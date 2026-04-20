"""
comparison_backbones.py — Faithful EEG backbone implementations for NeuroFlow comparison.

Architectures faithfully adapted from the BIOT repository (ycq091044/BIOT):
  https://github.com/ycq091044/BIOT

Each backbone is adapted to our data (26 channels, 3200 samples @ 200Hz) and
outputs a feature vector rather than class logits, for pairing with a DDPM head.

Models:
  SPaRCNet    : 1D DenseNet (Gemein et al. 2020 / BIOT paper baseline)
  ContraWR    : STFT + 2D ResBlocks (Yang et al. 2023 JMIR AI / BIOT baseline)
  FFCL        : STFT + CNN + LSTM dual-branch (BIOT baseline)
  cnn_trans   : STFT + CNN-Transformer (BIOT baseline)
  stt_trans   : Channel-attention + PatchEmbed + Transformer (STT, BIOT baseline)
  biot        : BIOT-style CLS-token patch transformer (Yang et al. NeurIPS 2023)
  beira       : Interpretable temporal CNN (NeuroBOLT paper baseline)
  li2024      : Dual temporal+spectral CNN (NeuroBOLT paper baseline)
  labram      : LaBraM/NeuroBOLT backbone (loads pretrained ckpt if path given)

Input:  [B, 26, 3200]  (EEG, pre-normalized by /100)
Output: [B, feat_dim]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Shared 2D ResBlock (BIOT-style — used by ContraWR, FFCL, CNN-Trans.)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock2D(nn.Module):
    """2D residual block with optional BN + ELU. Stride reduces spatial dims by 2."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ELU(inplace=True)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + self.skip(x))


def _stft_batch(x, n_fft=200, hop_length=10, n_channels=26):
    """Per-channel STFT → magnitude spectrogram.
    x: [B, C, T]  →  out: [B, C, n_fft//2+1, T_frames]
    """
    B, C, T = x.shape
    win = torch.hann_window(n_fft, device=x.device)
    # process all channels as a batch
    xf = x.reshape(B * C, T)
    spec = torch.stft(xf, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=win,
                      center=True, normalized=True, onesided=True,
                      return_complex=True)
    mag = spec.abs()                                     # [B*C, F, t]
    return mag.reshape(B, C, mag.shape[1], mag.shape[2])  # [B, C, F, t]


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPaRCNet (1D DenseNet, faithful to BIOT/PyHealth implementation)
#    Gemein et al. 2020; in_channels=16→26, sample_length=2000→3200
# ─────────────────────────────────────────────────────────────────────────────

class _DenseLayer1D(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_ch), nn.ELU(inplace=True),
            nn.Conv1d(in_ch, growth_rate, kernel_size=3, padding=1, bias=False),
        )
    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class _DenseBlock1D(nn.Module):
    def __init__(self, in_ch, growth_rate, n_layers=4):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(_DenseLayer1D(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_ch  = ch
    def forward(self, x):
        return self.layers(x)


class _Transition1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_ch), nn.ELU(inplace=True),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool1d(2),
        )
    def forward(self, x):
        return self.net(x)


class SPaRCNetBackbone(nn.Module):
    """
    SPaRCNet: 1D DenseNet-based temporal EEG encoder.
    Faithfully adapted from ycq091044/BIOT sparcnet.py.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 block_layers=4, growth_rate=16, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        # Number of dense blocks = log2(seq_len) - 1
        import math as _m
        n_blocks = int(_m.log2(seq_len)) - 1  # 11 for seq_len=3200 (log2=11.6)

        # Initial conv: reduce channels & downsample
        self.init_conv = nn.Sequential(
            nn.Conv1d(n_channels, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(2 * growth_rate), nn.ELU(inplace=True),
            nn.MaxPool1d(2),
        )
        ch = 2 * growth_rate

        # Dense blocks + transitions
        self.dense_blocks  = nn.ModuleList()
        self.transitions   = nn.ModuleList()
        for i in range(n_blocks):
            blk = _DenseBlock1D(ch, growth_rate, block_layers)
            self.dense_blocks.append(blk)
            ch = blk.out_ch
            if i < n_blocks - 1:
                out_ch = ch // 2
                self.transitions.append(_Transition1D(ch, out_ch))
                ch = out_ch
            else:
                self.transitions.append(nn.Identity())

        # Pool and project to feat_dim
        self.pool = nn.Sequential(
            nn.BatchNorm1d(ch), nn.ELU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch, feat_dim))

    def forward(self, x):
        h = self.init_conv(x)
        for blk, tr in zip(self.dense_blocks, self.transitions):
            h = tr(blk(h))
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ContraWR (STFT + 2D ResBlocks, faithful to BIOT contrawr.py)
#    Yang et al. 2021/2023 JMIR AI
# ─────────────────────────────────────────────────────────────────────────────

class ContraWRBackbone(nn.Module):
    """
    ContraWR encoder: STFT magnitude → 4× 2D ResBlocks → feature vector.
    Faithfully adapted from ycq091044/BIOT contrawr.py.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 n_fft=200, steps=20, dropout=0.1):
        super().__init__()
        self.feat_dim   = feat_dim
        self.n_fft      = n_fft
        self.hop_length = n_fft // steps

        # 4 × 2D ResBlocks: n_channels → 32 → 64 → 128 → 256
        self.conv1 = ResBlock2D(n_channels, 32,  stride=2)
        self.conv2 = ResBlock2D(32,         64,  stride=2)
        self.conv3 = ResBlock2D(64,         128, stride=2)
        self.conv4 = ResBlock2D(128,        256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(256, feat_dim) if feat_dim != 256 else nn.Identity()

    def forward(self, x):
        mag = _stft_batch(x, self.n_fft, self.hop_length)  # [B, C, F, t]
        h = self.conv1(mag)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)                    # [B, 256, F', t']
        h = self.pool(h).flatten(1)          # [B, 256]
        return self.proj(self.drop(h))


# ─────────────────────────────────────────────────────────────────────────────
# 3. FFCL (STFT+CNN + LSTM dual-branch, faithful to BIOT ffcl.py)
# ─────────────────────────────────────────────────────────────────────────────

class FFCLBackbone(nn.Module):
    """
    FFCL: Dual-branch — (1) STFT + 2D CNN, (2) shrunk temporal LSTM.
    Faithfully adapted from ycq091044/BIOT ffcl.py.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 n_fft=200, steps=20, shrink_steps=20, dropout=0.1):
        super().__init__()
        self.feat_dim   = feat_dim
        self.n_fft      = n_fft
        self.hop_length = n_fft // steps
        self.shrink_steps = shrink_steps

        # Spectral branch (same as ContraWR)
        self.conv1 = ResBlock2D(n_channels, 32,  stride=2)
        self.conv2 = ResBlock2D(32,         64,  stride=2)
        self.conv3 = ResBlock2D(64,         128, stride=2)
        self.conv4 = ResBlock2D(128,        256, stride=2)
        self.spec_pool = nn.AdaptiveAvgPool2d(1)

        # Temporal (LSTM) branch
        shrunk_len = seq_len // shrink_steps  # 3200//20 = 160
        self.lstm  = nn.LSTM(input_size=shrunk_len, hidden_size=256,
                             num_layers=2, batch_first=True, dropout=0.5)

        # Fusion: 256 + 256 → feat_dim
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(512, feat_dim)

    def forward(self, x):
        B, C, T = x.shape

        # Spectral branch
        mag = _stft_batch(x, self.n_fft, self.hop_length)  # [B, C, F, t]
        h1  = self.conv1(mag); h1 = self.conv2(h1); h1 = self.conv3(h1); h1 = self.conv4(h1)
        h1  = self.spec_pool(h1).flatten(1)                # [B, 256]

        # Temporal branch: shorten along time, use channel as "sequence"
        h2 = x[:, :, ::self.shrink_steps]  # stride → [B, C=26, T//shrink_steps=160]
        h2, _ = self.lstm(h2)              # [B, C, 256]
        h2  = h2[:, -1]                    # [B, 256] last hidden

        return self.proj(self.drop(torch.cat([h1, h2], dim=1)))


# ─────────────────────────────────────────────────────────────────────────────
# 4. CNN-Transformer (faithful to BIOT cnn_transformer.py)
# ─────────────────────────────────────────────────────────────────────────────

class CNNTransformerBackbone(nn.Module):
    """
    CNN-Transformer: segment-wise STFT+CNN → Transformer encoder.
    Faithfully adapted from ycq091044/BIOT cnn_transformer.py.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 n_fft=200, steps=20, n_segments=5,
                 nhead=4, n_tf_layers=4, dropout=0.2):
        super().__init__()
        self.feat_dim   = feat_dim
        self.n_fft      = n_fft
        self.hop_length = n_fft // steps
        self.n_segments = n_segments
        self.emb_size   = feat_dim

        # CNN feature extractor per segment (same as ContraWR)
        self.conv1 = ResBlock2D(n_channels, 32,  stride=2)
        self.conv2 = ResBlock2D(32,         64,  stride=2)
        self.conv3 = ResBlock2D(64,         128, stride=2)
        self.conv4 = ResBlock2D(128,        256, stride=2)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Project 256 → feat_dim if needed
        self.cnn_proj = nn.Linear(256, feat_dim) if feat_dim != 256 else nn.Identity()

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, n_segments, feat_dim))

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_tf_layers)
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        B, C, T = x.shape
        seg_len = T // self.n_segments

        # Process each segment
        segs = []
        for i in range(self.n_segments):
            xi = x[:, :, i * seg_len:(i + 1) * seg_len]    # [B, C, seg_len]
            mi = _stft_batch(xi, self.n_fft, self.hop_length)
            h  = self.conv1(mi); h = self.conv2(h); h = self.conv3(h); h = self.conv4(h)
            h  = self.cnn_pool(h).flatten(1)                # [B, 256]
            segs.append(self.cnn_proj(h))                   # [B, feat_dim]

        h = torch.stack(segs, dim=1)                        # [B, n_segs, feat_dim]
        h = h + self.pos_emb
        h = self.transformer(h)
        h = self.norm(h).mean(dim=1)                        # [B, feat_dim]
        return self.drop(h)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ST-Transformer (faithful to BIOT st_transformer.py)
# ─────────────────────────────────────────────────────────────────────────────

class _ChannelAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.attn = nn.Linear(n_channels, n_channels)
    def forward(self, x):
        # x: [B, C, T] → attention weights along C
        w = self.attn(x.mean(-1)).softmax(-1)  # [B, C]
        return x * w.unsqueeze(-1)


class _PatchSTEmbedding(nn.Module):
    def __init__(self, emb_size, n_channels, patch_size=200):
        super().__init__()
        self.proj = nn.Conv1d(n_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, T] → [B, T//patch_size, emb_size]
        return self.proj(x).transpose(1, 2)


class STTransformerBackbone(nn.Module):
    """
    ST-Transformer: ChannelAttention → PatchEmbed → TransformerEncoder.
    Faithfully adapted from ycq091044/BIOT st_transformer.py.
    Reference: arxiv 2106.11170 (EEG-Transformer, Song et al. 2021)
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 patch_size=200, depth=3, nhead=8, dropout=0.5):
        super().__init__()
        self.feat_dim = feat_dim

        self.chan_attn     = _ChannelAttention(n_channels)
        self.chan_norm     = nn.LayerNorm(seq_len)
        self.patch_embed   = _PatchSTEmbedding(feat_dim, n_channels, patch_size)

        n_patches = seq_len // patch_size                  # 16
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, feat_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(feat_dim)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        # Channel attention (residual)
        x = x + self.chan_attn(self.chan_norm(x))

        h = self.patch_embed(x)            # [B, n_patches, feat_dim]
        h = h + self.pos_emb
        h = self.transformer(h)
        return self.norm(h).mean(dim=1)   # [B, feat_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 6. BIOT (CLS-token patch transformer, Yang et al. NeurIPS 2023)
# ─────────────────────────────────────────────────────────────────────────────

class BIOTBackbone(nn.Module):
    """
    BIOT: flatten channel×patch tokens → CLS-token transformer.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 patch_size=200, n_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.feat_dim   = feat_dim
        self.patch_size = patch_size
        n_patches = seq_len // patch_size  # 16
        n_tokens  = n_channels * n_patches  # 416

        self.tokenizer = nn.Linear(patch_size, feat_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_tokens + 1, feat_dim))
        self.pos_drop   = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(feat_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, T = x.shape
        n = T // self.patch_size
        h = rearrange(x[:, :, :n * self.patch_size], 'B C (N P) -> B (C N) P', P=self.patch_size)
        h = self.tokenizer(h)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = self.pos_drop(h + self.pos_embed[:, :h.size(1)])
        h = self.transformer(h)
        return self.norm(h)[:, 0]          # CLS token


# ─────────────────────────────────────────────────────────────────────────────
# 7. BEIRA (interpretable CNN encoder, NeuroBOLT baseline)
# ─────────────────────────────────────────────────────────────────────────────

class BEIRABackbone(nn.Module):
    """
    BEIRA: temporal CNN encoder with channel attention.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64,      kernel_size=25, padding=12), nn.BatchNorm1d(64),      nn.ELU(),
            nn.Conv1d(64,        64,       kernel_size=11, padding=5),  nn.BatchNorm1d(64),      nn.ELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64,        128,      kernel_size=9,  padding=4),  nn.BatchNorm1d(128),     nn.ELU(),
            nn.Conv1d(128,       128,      kernel_size=5,  padding=2),  nn.BatchNorm1d(128),     nn.ELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128,       feat_dim, kernel_size=3,  padding=1),  nn.BatchNorm1d(feat_dim),nn.ELU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4), nn.ReLU(),
            nn.Linear(feat_dim // 4, feat_dim), nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h   = self.temporal(x)
        ctx = h.mean(-1)
        h   = h * self.attn(ctx).unsqueeze(-1)
        return self.drop(self.pool(h).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Li et al. 2024 (dual temporal+spectral CNN, NeuroBOLT baseline avg R=0.445)
# ─────────────────────────────────────────────────────────────────────────────

class LiEtAl2024Backbone(nn.Module):
    """
    Li et al. 2024 EEG-to-fMRI encoder: spectral + temporal CNN fusion.
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 n_fft=128, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_fft    = n_fft

        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,        128,  kernel_size=7,  stride=2, padding=3), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )  # → [B, 128, 32]

        self.spectral_branch = nn.Sequential(
            nn.Conv2d(n_channels, 64,  kernel_size=(3,3), padding=(1,1)), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,        128,  kernel_size=(3,3), stride=(2,2), padding=(1,1)), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )  # → [B, 128, 4, 8]

        t_flat = 128 * 32
        s_flat = 128 * 4 * 8
        self.fusion = nn.Sequential(
            nn.Linear(t_flat + s_flat, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, feat_dim),
        )

    def forward(self, x):
        B, C, T = x.shape
        t_feat  = self.temporal_branch(x).flatten(1)

        xf   = x.reshape(B * C, T)
        win  = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(xf, n_fft=self.n_fft, hop_length=self.n_fft//4,
                          window=win, return_complex=True)
        mag  = stft.abs().reshape(B, C, stft.shape[1], stft.shape[2])
        s_feat = self.spectral_branch(mag).flatten(1)

        return self.fusion(torch.cat([t_feat, s_feat], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 9. LaBraM/NeuroBOLT backbone wrapper (uses pretrained ckpt if available)
# ─────────────────────────────────────────────────────────────────────────────

class NeuroBOLTSingleBackbone(nn.Module):
    """
    Single (non-ROI-specific) NeuroBOLT backbone → 200D features.
    Load from ANY of the 7 ROI-specific checkpoints; freeze params.
    Used as NeuroBOLT baseline (single backbone, not 7-specialized).
    """
    def __init__(self, ckpt_path=None, n_channels=26, feat_dim=200, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self._ckpt_path = ckpt_path
        self._loaded = False

    def load(self, device='cpu'):
        """Lazy-load NeuroBOLT backbone from checkpoint."""
        import sys, os
        sys.path.insert(0, os.path.join(
            'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm', 'code'))
        from timm.models import create_model
        import models.model
        m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                         drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                         drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                         use_rel_pos_bias=True, use_abs_pos_emb=True,
                         init_values=0.1, qkv_bias=True)
        if self._ckpt_path and os.path.exists(self._ckpt_path):
            st = torch.load(self._ckpt_path, map_location='cpu', weights_only=False)
            st = st.get('model', st)
            for k in ['head.weight', 'head.bias']:
                if k in st and st[k].shape != m.state_dict()[k].shape:
                    del st[k]
            for k in list(st.keys()):
                if 'relative_position_index' in k:
                    st.pop(k)
            m.load_state_dict(st, strict=False)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m.to(device)
        self._loaded = True

    @property
    def input_chans(self):
        import sys, os
        sys.path.insert(0, os.path.join(
            'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm', 'code'))
        from utils import get_input_chans
        CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
                    'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
                    'POZ','OZ','FT9','FT10','TP9','TP10']
        return get_input_chans(CH_NAMES)

    def forward(self, x):
        # x: [B, 26, 3200]
        from einops import rearrange as _r
        b = _r(x, 'B N (A T) -> B N A T', T=200)
        ic = self.input_chans
        xt = self.model.forward_ts_features(b, input_chans=ic)
        xm = self.model.mss_module(_r(b, 'B N A T -> B N (A T)'), input_chans=None)
        return self.model.head_act(xm + xt)          # [B, 200]


# ─────────────────────────────────────────────────────────────────────────────
# 10. LaBraM-style backbone (spectral tokenizer + Transformer)
#     Inspired by Jiang et al. 2024 "Large Brain Model" — trainable version.
#     Architecture: FFT patch tokenizer → CLS-token Transformer → feat_dim
# ─────────────────────────────────────────────────────────────────────────────

class LaBraMBackbone(nn.Module):
    """
    LaBraM-style EEG backbone (trainable from scratch).
    Input: [B, C, T]
    Steps:
      1. Divide T into n_patches patches of patch_size samples
      2. Per-channel FFT per patch → frequency magnitude [C * n_patches, freq_dim]
      3. Linear projection → emb_dim
      4. Add learnable channel + patch position embeddings
      5. Transformer encoder with CLS token
      6. CLS output → Linear → feat_dim
    """
    def __init__(self, n_channels=26, seq_len=3200, feat_dim=256,
                 patch_size=200, emb_dim=256, n_heads=8, n_layers=6,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.feat_dim  = feat_dim
        self.n_channels = n_channels
        self.patch_size = patch_size
        n_patches  = seq_len // patch_size           # 16 patches of 200 samples
        freq_dim   = patch_size // 2 + 1             # 101 freq bins

        # Token projection: FFT magnitude → embedding
        self.token_proj = nn.Sequential(
            nn.Linear(freq_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        # Learnable embeddings: channel-wise + patch-wise (additive, like LaBraM)
        self.chan_emb  = nn.Embedding(n_channels, emb_dim)
        self.patch_emb = nn.Embedding(n_patches,  emb_dim)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, feat_dim)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        P = self.patch_size
        n_patches = T // P
        # Split into patches: [B, C, n_patches, P]
        xp = x[:, :, :n_patches * P].reshape(B, C, n_patches, P)
        # FFT per patch: [B*C*n_patches, P] → [B*C*n_patches, P//2+1]
        xp_flat = xp.reshape(B * C * n_patches, P)
        freq = torch.fft.rfft(xp_flat, norm='ortho').abs()  # [B*C*np, F]
        # Project to embedding: [B, C, n_patches, emb_dim]
        tokens = self.token_proj(freq).reshape(B, C, n_patches, -1)
        # Add channel and patch embeddings
        c_idx = torch.arange(C, device=x.device)
        p_idx = torch.arange(n_patches, device=x.device)
        tokens = tokens + self.chan_emb(c_idx).unsqueeze(1)   # broadcast over patches
        tokens = tokens + self.patch_emb(p_idx).unsqueeze(0)  # broadcast over channels
        # Flatten to sequence: [B, C*n_patches, emb_dim]
        tokens = tokens.reshape(B, C * n_patches, -1)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # Transformer
        out = self.transformer(tokens)
        out = self.norm(out)
        return self.head(out[:, 0])  # CLS token → [B, feat_dim]


# ─────────────────────────────────────────────────────────────────────────────
# REVE backbone (NeurIPS 2025) — pretrained from HuggingFace brain-bzh/reve-base
# ─────────────────────────────────────────────────────────────────────────────

class REVEBackbone(nn.Module):
    """
    REVE: large-scale pretrained EEG foundation model (NeurIPS 2025).
    60k+ hours pretraining, 4D positional encoding.
    Frozen encoder, mean-pool over token sequence → [B, 512].
    """
    REVE_REPO = 'C:/Users/PC/reve_eeg/src'
    HF_MODEL  = 'brain-bzh/reve-base'
    CH_NAMES  = ['Fp1','F3','F7','FC3','C3','C5','P3','P7','P9','PO7','PO3','O1',
                 'Oz','Pz','CPz','Fp2','Fz','F4','F8','FC4','FCz','C4','C6','P4',
                 'P8','P10']

    def __init__(self, n_channels=26, seq_len=3200, feat_dim=512, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self._loaded  = False
        # Pre-compute and store normalized MNI electrode positions [C, 3]
        pos = self._get_mni_positions()  # [26, 3] numpy
        self.register_buffer('_pos', torch.tensor(pos, dtype=torch.float32))

    @staticmethod
    def _get_mni_positions():
        """Return normalized MNI xyz positions [26, 3] for CH_NAMES."""
        import mne
        import numpy as np
        ch_names = REVEBackbone.CH_NAMES
        mont = mne.channels.make_standard_montage('standard_1020')
        ch_pos = mont.get_positions()['ch_pos']
        xyz = np.array([ch_pos[ch] for ch in ch_names], dtype=np.float32)  # [26, 3]
        # Normalize: subtract mean, divide by sqrt(3 * mean_squared_dist)
        xyz -= xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz ** 2, axis=1)))
        xyz /= (scale + 1e-8)
        return xyz  # [26, 3]

    def load(self, device='cpu'):
        import sys, os
        # Set HF_TOKEN in your shell / environment before running.
        # REVE is a gated HuggingFace model; a valid classic-read token with
        # "Access public gated repositories" permission is required.
        if 'HF_TOKEN' not in os.environ:
            raise RuntimeError(
                'HF_TOKEN environment variable not set. '
                'Obtain a classic read token from HuggingFace (with access to '
                "brain-bzh/reve-base) and export it before running."
            )
        sys.path.insert(0, self.REVE_REPO)
        from models.encoder import REVE
        model, _ = REVE.from_pretrained(self.HF_MODEL)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device)
        self._loaded = True
        print(f'  [REVE] loaded {self.HF_MODEL} → {device}', flush=True)

    def forward(self, x):  # x: [B, 26, 3200]
        device = x.device
        if not self._loaded:
            self.load(device)
        elif next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

        B = x.shape[0]
        pos = self._pos.unsqueeze(0).expand(B, -1, -1).to(device)  # [B, 26, 3]
        with torch.no_grad():
            tokens = self.model(x, pos)          # [B, C*patches, 512]
        feat = tokens.mean(dim=1)                # [B, 512]
        if self.feat_dim != 512:
            # linear projection if caller requests different dim (shouldn't happen)
            if not hasattr(self, '_proj'):
                self._proj = nn.Linear(512, self.feat_dim).to(device)
            feat = self._proj(feat)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# BrainOmni backbone (NeurIPS 2025) — pretrained from OpenTSLab/BrainOmni
# ─────────────────────────────────────────────────────────────────────────────

class BrainOmniBackbone(nn.Module):
    """
    BrainOmni: EEG+MEG foundation model with VQ-VAE tokenizer (NeurIPS 2025).
    Pretrained on 2600+ hours of EEG and MEG.
    Frozen encoder, mean-pool over channels and windows → [B, lm_dim=512].
    """
    BRAINOMNI_ROOT  = 'C:/Users/PC/BrainOmni'
    CKPT_BASE       = 'C:/Users/PC/BrainOmni/ckpt_collection/base'
    CKPT_TOKENIZER  = 'C:/Users/PC/BrainOmni/ckpt_collection/braintokenizer'
    CH_NAMES        = ['Fp1','F3','F7','FC3','C3','C5','P3','P7','P9','PO7','PO3','O1',
                       'Oz','Pz','CPz','Fp2','Fz','F4','F8','FC4','FCz','C4','C6','P4',
                       'P8','P10']

    def __init__(self, n_channels=26, seq_len=3200, feat_dim=512, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self._loaded  = False
        # Pre-compute normalized 6D positions [26, 6] (EEG: xyz + [0,0,0])
        pos, stype = self._get_eeg_positions()
        self.register_buffer('_pos',         torch.tensor(pos,   dtype=torch.float32))
        self.register_buffer('_sensor_type', torch.tensor(stype, dtype=torch.int32))

    @staticmethod
    def _get_eeg_positions():
        """Return normalized 6D positions [26, 6] and sensor_type [26] for EEG."""
        import mne
        import numpy as np
        ch_names = BrainOmniBackbone.CH_NAMES
        mont = mne.channels.make_standard_montage('standard_1020')
        ch_pos = mont.get_positions()['ch_pos']
        xyz = np.array([ch_pos[ch] for ch in ch_names], dtype=np.float32)  # [26, 3]
        # Normalize (same as BrainOmni factory/utils.py normalize_pos for EEG)
        xyz -= xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz ** 2, axis=1)))
        xyz /= (scale + 1e-8)
        # EEG: last 3 dims are zeros (direction is zero for EEG, only used for MEG)
        pos = np.concatenate([xyz, np.zeros_like(xyz)], axis=1)  # [26, 6]
        sensor_type = np.zeros(len(ch_names), dtype=np.int32)    # 0 = EEG
        return pos, sensor_type

    def load(self, device='cpu'):
        import sys, json, os
        sys.path.insert(0, self.BRAINOMNI_ROOT)
        from brainomni.model import BrainOmni

        with open(os.path.join(self.CKPT_BASE, 'model_cfg.json')) as f:
            cfg = json.load(f)
        model = BrainOmni(**cfg)
        ckpt = torch.load(os.path.join(self.CKPT_BASE, 'BrainOmni.pt'),
                          map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt, strict=False)

        # Always freeze tokenizer (Stage 1) — per official recommendation
        for p in model.tokenizer.parameters():
            p.requires_grad_(False)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        # Manually load tokenizer weights (they're separate)
        tok_ckpt = torch.load(os.path.join(self.CKPT_TOKENIZER, 'BrainTokenizer.pt'),
                              map_location='cpu', weights_only=True)
        model.tokenizer.load_state_dict(tok_ckpt, strict=False)

        self.model   = model.to(device)
        self.lm_dim  = cfg['lm_dim']   # 512 for base
        self._loaded = True
        print(f'  [BrainOmni] loaded base → {device}', flush=True)

    def forward(self, x):  # x: [B, 26, 3200]
        device = x.device
        if not self._loaded:
            self.load(device)
        elif next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

        B = x.shape[0]
        pos  = self._pos.unsqueeze(0).expand(B, -1, -1).to(device)       # [B, 26, 6]
        stype = self._sensor_type.unsqueeze(0).expand(B, -1).to(device)  # [B, 26]

        with torch.no_grad():
            feats = self.model.encode(x, pos, stype)   # [B, 26, W, lm_dim]

        feat = feats.mean(dim=(1, 2))  # [B, lm_dim=512]
        if self.feat_dim != self.lm_dim:
            if not hasattr(self, '_proj'):
                self._proj = nn.Linear(self.lm_dim, self.feat_dim).to(device)
            feat = self._proj(feat)
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    'sparc'      : SPaRCNetBackbone,
    'contrawr'   : ContraWRBackbone,
    'ffcl'       : FFCLBackbone,
    'cnn_trans'  : CNNTransformerBackbone,
    'stt_trans'  : STTransformerBackbone,
    'biot'       : BIOTBackbone,
    'beira'      : BEIRABackbone,
    'li2024'     : LiEtAl2024Backbone,
    'neurobolt'  : NeuroBOLTSingleBackbone,
    'labram'     : LaBraMBackbone,
    'reve'       : REVEBackbone,
    'brainomni'  : BrainOmniBackbone,
}

# Paper table display names & reference performance (from NeuroBOLT paper)
BACKBONE_DISPLAY = {
    'sparc'      : 'SPaRCNet+DDPM',
    'contrawr'   : 'ContraWR+DDPM',
    'ffcl'       : 'FFCL+DDPM',
    'cnn_trans'  : 'CNN-Trans.+DDPM',
    'stt_trans'  : 'STT-Trans.+DDPM',
    'biot'       : 'BIOT+DDPM',
    'beira'      : 'BEIRA+DDPM',
    'li2024'     : 'Li et al.+DDPM',
    'neurobolt'  : 'NeuroBOLT+DDPM',
    'neuroflow'  : 'NeuroFlow (Ours)',
    'labram'     : 'LaBraM-style+DDPM',
    'reve'       : 'REVE+DDPM',
    'brainomni'  : 'BrainOmni+DDPM',
}

# Inter-subject R from NeuroBOLT paper (with LINEAR head) — reference only
PAPER_INTER_R = {
    'ffcl'     : 0.376,
    'cnn_trans': 0.273,
    'stt_trans': 0.218,
    'biot'     : 0.435,
    'beira'    : 0.412,
    'li2024'   : 0.419,  # "Li et al." from paper
}
# Intra-subject R from NeuroBOLT paper (LINEAR head)
PAPER_INTRA_R = {
    'biot'     : 0.473,
    'beira'    : 0.341,
    'li2024'   : 0.445,
}


def build_backbone(name, n_channels=26, seq_len=3200, feat_dim=256, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Options: {sorted(BACKBONE_REGISTRY.keys())}")
    cls = BACKBONE_REGISTRY[name]
    if name == 'neurobolt':
        return cls(n_channels=n_channels, feat_dim=200, **kwargs)
    if name in ('reve', 'brainomni'):
        # These pretrained models have fixed output dims (512); feat_dim arg is ignored
        return cls(n_channels=n_channels, seq_len=seq_len, feat_dim=512, **kwargs)
    return cls(n_channels=n_channels, seq_len=seq_len, feat_dim=feat_dim, **kwargs)
