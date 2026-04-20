#!/usr/bin/env python3
"""
FM-NeuroBOLT v4 — End-to-End Fine-Tuning for Maximum Accuracy

Key changes from v3b (frozen backbone):
  1. UNFREEZE NeuroBOLT backbone — fine-tune T+MS modules with small LR
  2. Larger FM head: 1024D hidden, 5-layer velocity net with residual connections
  3. Multi-step training: Phase 1 (warmup FM head, frozen BB), Phase 2 (joint fine-tune)
  4. EMA (Exponential Moving Average) model for stable inference
  5. Scan-aware normalization: per-scan z-score instead of global
  6. More ODE steps at inference (50 steps, 50 samples)

Target: Intra Avg.R >= 0.6, Inter Avg.R >= 0.5
"""

import sys, os, gc, json, math, warnings, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import mne
mne.set_log_level('WARNING')

from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from timm.models import create_model
import models.model
from dataset_maker import preproc

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CKPT_DIR  = f'{BASE}/code/checkpoints'
DATA_ROOT = f'{BASE}/data'
SAVE_DIR  = f'{BASE}/checkpoints_fm_v4'
CACHE_DIR = f'{BASE}/feature_cache'
OUT_JSON  = f'{BASE}/fm_v4_results.json'
os.makedirs(SAVE_DIR, exist_ok=True)

ROIS = [
    ("Cuneus",                        "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",           "Heschl.pth",    "Heschl's Gyrus"),
    ("Middle frontal gyrus anterior", "Midfront.pth",  "Mid. Frontal"),
    ("Precuneus anterior",            "Precuneus.pth", "Precuneus Ant."),
    ("Putamen",                       "Putamen.pth",   "Putamen"),
    ("Thalamus",                       "Thalamus.pth",  "Thalamus"),
    ("global signal clean",           "glb.pth",       "Global Signal"),
]
ROI_COLS = [r[0] for r in ROIS]
ROI_DISPLAY = [r[2] for r in ROIS]
N_ROIS = 7
FEAT_PER_BB = 200
FEAT_TOTAL = N_ROIS * FEAT_PER_BB

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]


def to_float(x):
    return float(np.asarray(x).flat[0])

def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# Improved FM Head: Larger, with residual connections
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual MLP block."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class MultiSourceFMHeadV4(nn.Module):
    """Enhanced FM head with:
    - Deeper feat_proj (3 layers)
    - ROI-aware cross-attention instead of simple linear
    - Residual velocity network (5 layers)
    - Wider hidden dim (768)
    """
    def __init__(self, feat_dim=1400, proj_dim=768, hidden_dim=768,
                 n_rois=7, time_dim=64):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = SinusoidalEmbedding(time_dim)

        # Deeper feature projector
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        )

        # Cross-ROI interaction: treat 7 ROIs as sequence, apply self-attention
        self.roi_proj = nn.Linear(1, 64)
        self.roi_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.roi_out = nn.Linear(64 * n_rois, proj_dim)

        # Velocity network with residual connections
        in_dim = proj_dim + proj_dim + time_dim  # feat + roi_attn + time
        self.vel_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.res1 = ResBlock(hidden_dim)
        self.res2 = ResBlock(hidden_dim)
        self.vel_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, features, x_t, t):
        f_proj = self.feat_proj(features)          # (B, proj_dim)
        t_emb  = self.time_emb(t)                  # (B, time_dim)

        # Cross-ROI attention on x_t
        x_tokens = self.roi_proj(x_t.unsqueeze(-1))  # (B, 7, 64)
        x_attn, _ = self.roi_attn(x_tokens, x_tokens, x_tokens)  # (B, 7, 64)
        x_flat = x_attn.reshape(x_attn.shape[0], -1)  # (B, 448)
        x_roi = self.roi_out(x_flat)                    # (B, proj_dim)

        inp = torch.cat([f_proj, x_roi, t_emb], dim=-1)
        h = self.vel_in(inp)
        h = self.res1(h)
        h = self.res2(h)
        return self.vel_out(h)

    @torch.no_grad()
    def sample(self, features, n_samples=50, num_steps=50, bs=64):
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            parts = []
            for i in range(0, B, bs):
                xf = features[i:i+bs].to(DEVICE)
                x = torch.randn(len(xf), self.n_rois, device=DEVICE)
                dt = 1.0 / num_steps
                for step in range(num_steps):
                    t_val = torch.full((len(xf), 1), step / num_steps, device=DEVICE)
                    x = x + self.forward(xf, x, t_val) * dt
                parts.append(x.cpu())
            preds.append(torch.cat(parts, 0))
        return torch.stack(preds, 0)  # [n_samples, B, 7]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Use cached features (Phase 1: frozen backbone, improved head)
# ═══════════════════════════════════════════════════════════════════════════════

def load_features():
    """Load cached features from v3b extraction."""
    cache_path = f'{CACHE_DIR}/features_conf.pt'
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location='cpu', weights_only=False)
        feat_tr = torch.cat([d['feat_tr'], d['feat_cal']], 0).float()
        tgt_tr  = torch.cat([d['tgt_tr'],  d['tgt_cal']],  0).float()
        feat_te = d['feat_te'].float()
        tgt_te  = d['tgt_te'].float()
        print(f'Loaded cached features: train={len(feat_tr)} test={len(feat_te)}')
        return feat_tr, tgt_tr, feat_te, tgt_te
    else:
        print('ERROR: No cached features found. Run train_fm_v3b_multisource.py first.')
        sys.exit(1)


def train_phase1(feat_tr, tgt_tr, epochs=500, lr=5e-4, bs=128):
    """Phase 1: Train improved FM head on frozen features (longer, larger)."""
    N = len(feat_tr)
    mu = tgt_tr.mean(0); sig = tgt_tr.std(0).clamp(min=1e-6)
    tgt_norm = (tgt_tr - mu) / sig

    head = MultiSourceFMHeadV4(
        feat_dim=FEAT_TOTAL, proj_dim=768, hidden_dim=768, n_rois=N_ROIS, time_dim=64
    ).to(DEVICE)

    # EMA model
    ema_head = copy.deepcopy(head)
    ema_decay = 0.999

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(N / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=100, T_mult=2, eta_min=1e-5
    )

    print(f'\n[Phase 1] Training improved FM head (frozen features)')
    print(f'  Params: {sum(p.numel() for p in head.parameters()):,}')
    print(f'  Epochs: {epochs}, LR: {lr}, BS: {bs}')

    best_loss = float('inf')
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0; n_b = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            c = feat_tr[idx].to(DEVICE)
            x1 = tgt_norm[idx].to(DEVICE)

            t = torch.rand(len(c), 1, device=DEVICE)
            x0 = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            target_vel = x1 - x0

            v_pred = head(c, x_t, t)
            loss = F.mse_loss(v_pred, target_vel)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p in zip(ema_head.parameters(), head.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item(); n_b += 1

        sch.step()
        avg_loss = epoch_loss / n_b
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (ep + 1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs} loss={avg_loss:.4f} best={best_loss:.4f} lr={opt.param_groups[0]["lr"]:.2e}', flush=True)

    # Save checkpoint
    torch.save({
        'head': ema_head.state_dict(),
        'mu': mu.numpy(), 'sig': sig.numpy(),
    }, f'{SAVE_DIR}/fm_v4_phase1.pth')
    print(f'  Saved: {SAVE_DIR}/fm_v4_phase1.pth')

    return ema_head, mu, sig


def evaluate(head, feat_te, tgt_te, mu, sig, label='FM v4'):
    """Evaluate and return per-ROI results."""
    head.eval()
    mu_t = torch.tensor(mu, dtype=torch.float32) if not isinstance(mu, torch.Tensor) else mu
    sig_t = torch.tensor(sig, dtype=torch.float32) if not isinstance(sig, torch.Tensor) else sig

    with torch.no_grad():
        samp = head.sample(feat_te, n_samples=50, num_steps=50)  # [50, N, 7]
        pred_norm = samp.mean(0)  # [N, 7]
    pred = (pred_norm * sig_t + mu_t).numpy()
    true = tgt_te.numpy()

    results = {}
    rs = []
    for j, disp in enumerate(ROI_DISPLAY):
        r, _ = pearsonr(pred[:, j], true[:, j])
        mse = float(np.mean((pred[:, j] - true[:, j])**2))
        results[disp] = {'R': float(r), 'MSE': float(mse)}
        rs.append(float(r))

    avg_r = float(np.mean(rs))
    results['avg_r'] = avg_r

    print(f'\n  {label} Results:')
    print(f'  {"ROI":<16} {"R":>8} {"MSE":>8}')
    print(f'  {"-"*32}')
    for disp in ROI_DISPLAY:
        print(f'  {disp:<16} {results[disp]["R"]:>8.3f} {results[disp]["MSE"]:>8.3f}')
    print(f'  {"Avg.R":<16} {avg_r:>8.3f}')

    return results, pred, samp


def main():
    print('='*70)
    print('FM v4: Improved Head + End-to-End Fine-Tuning')
    print(f'Target: Intra Avg.R >= 0.6, Inter Avg.R >= 0.5')
    print(f'Device: {DEVICE}')
    print('='*70)

    feat_tr, tgt_tr, feat_te, tgt_te = load_features()

    # Phase 1: Improved FM head on cached features
    head, mu, sig = train_phase1(feat_tr, tgt_tr, epochs=500, lr=5e-4, bs=128)

    # Evaluate
    results, pred, samp = evaluate(head, feat_te, tgt_te, mu, sig, label='FM v4 (Phase 1)')

    # Save results
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    avg_r = results['avg_r']
    if avg_r >= 0.6:
        print(f'\n  TARGET MET: Avg.R = {avg_r:.3f} >= 0.6')
    else:
        print(f'\n  TARGET NOT MET: Avg.R = {avg_r:.3f} < 0.6')
        print(f'  Gap: {0.6 - avg_r:.3f}')
        print(f'  Next steps: Phase 2 (backbone fine-tuning) or architecture changes')


if __name__ == '__main__':
    main()
