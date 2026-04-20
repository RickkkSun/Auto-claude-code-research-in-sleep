#!/usr/bin/env python3
"""
FM v6 — End-to-End Fine-Tuning of NeuroBOLT Backbone + FM Head

Key insight: NeuroBOLT is only 7.9M params per backbone (not 86M).
7 backbones = 55M params total — easily fits on 16GB GPU.

Strategy:
  Phase 1: Warm up FM head with frozen backbones (100 epochs)
  Phase 2: Unfreeze backbone last layers + joint fine-tune (200 epochs, small LR)
  Phase 3: Full fine-tune everything (100 epochs, very small LR)

Target: Intra Avg.R >= 0.6
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
SAVE_DIR  = f'{BASE}/checkpoints_fm_v6'
OUT_JSON  = f'{BASE}/fm_v6_results.json'
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
# End-to-End Model: 7 Backbones + FM Head
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        return torch.cat([(t * freqs).sin(), (t * freqs).cos()], dim=-1)


class MultiSourceFMHead(nn.Module):
    """Same proven v3b head architecture."""
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512,
                 n_rois=7, time_dim=32):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        )
        self.roi_interact = nn.Linear(n_rois, n_rois)
        in_dim = proj_dim + n_rois + time_dim
        self.velocity_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_rois),
        )

    def forward(self, features, x_t, t):
        f_proj = self.feat_proj(features)
        t_emb  = self.time_emb(t)
        x_t_i  = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f_proj, x_t_i, t_emb], dim=-1))

    @torch.no_grad()
    def sample(self, features, n_samples=50, num_steps=50, bs=64):
        B = features.shape[0]; preds = []
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
        return torch.stack(preds, 0)


class E2EModel(nn.Module):
    """End-to-end: 7 NeuroBOLT backbones → concat features → FM head."""
    def __init__(self, backbones, fm_head):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.fm_head = fm_head
        self.input_chans = get_input_chans()

    def extract_features(self, eeg_batch):
        """eeg_batch: (B, 26, 3200) raw EEG."""
        eeg_4d = rearrange(eeg_batch, 'B N (A T) -> B N A T', T=200)
        feats = []
        for bb in self.backbones:
            xt = bb.forward_ts_features(eeg_4d, input_chans=self.input_chans)
            xm = bb.mss_module(eeg_batch, input_chans=None)
            feats.append(bb.head_act(xm + xt))  # (B, 200)
        return torch.cat(feats, dim=1)  # (B, 1400)

    def forward(self, eeg_batch, x_t, t):
        """Full forward: EEG → features → FM velocity."""
        features = self.extract_features(eeg_batch)
        return self.fm_head(features, x_t, t)

    def freeze_backbones(self):
        for bb in self.backbones:
            for p in bb.parameters():
                p.requires_grad_(False)

    def unfreeze_backbone_last_layers(self, n_blocks=2):
        """Unfreeze last N transformer blocks + mss_module (float tensors only)."""
        for bb in self.backbones:
            total_blocks = len(bb.blocks)
            for i in range(total_blocks - n_blocks, total_blocks):
                for p in bb.blocks[i].parameters():
                    if p.is_floating_point():
                        p.requires_grad_(True)
            for p in bb.mss_module.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)
            if hasattr(bb, 'fc_norm'):
                for p in bb.fc_norm.parameters():
                    if p.is_floating_point():
                        p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)


def build_backbone(ckpt_path):
    m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                     drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=True, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=True)
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = st['model'] if 'model' in st else st
    for k in ['head.weight', 'head.bias']:
        if k in st and st[k].shape != m.state_dict()[k].shape:
            del st[k]
    for k in list(st.keys()):
        if 'relative_position_index' in k:
            st.pop(k)
    m.load_state_dict(st, strict=False)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading (raw EEG + fMRI targets)
# ═══════════════════════════════════════════════════════════════════════════════

def load_scan_all_rois(sub, scan):
    patient = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200: raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    df = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15 / (0.5 / TR), btype='low')

    all_fmri = []; eeg_all = None; n = None
    for col in ROI_COLS:
        try:
            fm = df[[col]].to_numpy().T
        except KeyError:
            del raw; return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            n = len(ep["eeg"])
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ep["eeg"]])
        all_fmri.append(torch.tensor([to_float(x) for x in ep["fmri"]], dtype=torch.float32))

    del raw
    fmri_all = torch.stack(all_fmri, dim=1)
    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / TR)

    return (eeg_all[:valcrop], fmri_all[:valcrop],
            eeg_all[valcrop:], fmri_all[valcrop:])


def load_all_data():
    print('[Data] Loading all scans...')
    eeg_tr_list, tgt_tr_list = [], []
    eeg_te_list, tgt_te_list = [], []

    for sub, scan in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'  {pat}...', end=' ', flush=True)
        try:
            result = load_scan_all_rois(sub, scan)
            if result is None:
                print('SKIP'); continue
            et, tt, ete, tte = result
            if len(ete) < 5:
                print('SKIP (short)'); continue
            eeg_tr_list.append(et)
            tgt_tr_list.append(tt)
            eeg_te_list.append(ete)
            tgt_te_list.append(tte)
            print(f'tr={len(et)} te={len(ete)}')
        except Exception as e:
            print(f'SKIP ({e})')

    eeg_tr = torch.cat(eeg_tr_list, 0)
    tgt_tr = torch.cat(tgt_tr_list, 0)
    eeg_te = torch.cat(eeg_te_list, 0)
    tgt_te = torch.cat(tgt_te_list, 0)
    print(f'  Total: train={len(eeg_tr)} test={len(eeg_te)}')
    return eeg_tr, tgt_tr, eeg_te, tgt_te


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_phase(model, eeg_tr, tgt_tr_norm, epochs, lr, bs, phase_name):
    """Generic training phase."""
    N = len(eeg_tr)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'\n  [{phase_name}] Trainable: {n_train:,}/{n_total:,} params, '
          f'Epochs: {epochs}, LR: {lr}', flush=True)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0; n_b = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            eeg = (eeg_tr[idx] / 100.0).to(DEVICE)  # scale EEG
            x1 = tgt_tr_norm[idx].to(DEVICE)

            t = torch.rand(len(eeg), 1, device=DEVICE)
            x0 = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            target_vel = x1 - x0

            v_pred = model(eeg, x_t, t)
            loss = F.mse_loss(v_pred, target_vel)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            epoch_loss += loss.item(); n_b += 1

        sch.step()
        if (ep + 1) % 25 == 0:
            print(f'    ep {ep+1}/{epochs} loss={epoch_loss/n_b:.4f} '
                  f'lr={opt.param_groups[0]["lr"]:.2e}', flush=True)


def evaluate_e2e(model, eeg_te, tgt_te, mu, sig, label='E2E'):
    model.eval()
    N = len(eeg_te)
    bs = 32

    # Extract features first (memory-efficient)
    feat_parts = []
    with torch.no_grad():
        for i in range(0, N, bs):
            eeg = (eeg_te[i:i+bs] / 100.0).to(DEVICE)
            feat_parts.append(model.extract_features(eeg).cpu())
    features = torch.cat(feat_parts, 0)

    # Sample using FM head
    with torch.no_grad():
        samp = model.fm_head.sample(features, n_samples=20, num_steps=50)
        pred_norm = samp.mean(0)
    pred = (pred_norm * sig + mu).numpy()
    true = tgt_te.numpy()

    results = {}; rs = []
    for j, disp in enumerate(ROI_DISPLAY):
        r, _ = pearsonr(pred[:, j], true[:, j])
        results[disp] = {'R': float(r)}
        rs.append(float(r))

    avg_r = float(np.mean(rs))
    results['avg_r'] = avg_r

    print(f'\n  {label}:')
    print(f'  | {"ROI":<16} | {"R":>6} |')
    print(f'  |{"-"*18}|{"-"*8}|')
    for disp in ROI_DISPLAY:
        print(f'  | {disp:<16} | {results[disp]["R"]:>6.3f} |')
    print(f'  | {"Avg.R":<16} | {avg_r:>6.3f} |')

    return results


def main():
    print('='*70)
    print('FM v6: End-to-End Fine-Tuning')
    print(f'Target: Intra Avg.R >= 0.6')
    print(f'Device: {DEVICE}')
    print('='*70)

    # Load raw data
    eeg_tr, tgt_tr, eeg_te, tgt_te = load_all_data()

    # Normalize targets
    mu = tgt_tr.mean(0); sig = tgt_tr.std(0).clamp(min=1e-6)
    tgt_tr_norm = (tgt_tr - mu) / sig

    # Build model
    print('\n[Building E2E model]')
    backbones = []
    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'  Loading backbone {j+1}/7: {display}...', flush=True)
        bb = build_backbone(ckpt_path)
        backbones.append(bb)

    # Load pretrained FM head from v3b
    fm_head = MultiSourceFMHead(feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=N_ROIS)
    v3b_ckpt = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
    if os.path.exists(v3b_ckpt):
        ckpt = torch.load(v3b_ckpt, map_location='cpu', weights_only=False)
        state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
        fm_head.load_state_dict(ckpt[state_key] if state_key else ckpt, strict=True)
        print('  Loaded pretrained FM head from v3b')

    model = E2EModel(backbones, fm_head).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total model params: {total_params:,} ({total_params*4/1e6:.1f} MB)')

    # Phase 1: Freeze backbones, warm up FM head (quick)
    print('\n' + '='*70)
    print('Phase 1: FM head warmup (frozen backbones)')
    model.freeze_backbones()
    phase1_ckpt = f'{SAVE_DIR}/fm_v6_phase1.pth'
    if os.path.exists(phase1_ckpt):
        print(f'  Loading Phase 1 checkpoint from {phase1_ckpt}')
        ckpt_data = torch.load(phase1_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt_data['model'], strict=False)
        mu = torch.tensor(ckpt_data['mu'])
        sig = torch.tensor(ckpt_data['sig'])
        tgt_tr_norm = (tgt_tr - mu) / sig
        r1 = evaluate_e2e(model, eeg_te, tgt_te, mu, sig, 'Phase 1 (loaded)')
    else:
        train_phase(model, eeg_tr, tgt_tr_norm, epochs=200, lr=3e-4, bs=32, phase_name='Phase1')
        r1 = evaluate_e2e(model, eeg_te, tgt_te, mu, sig, 'Phase 1 (frozen BB)')
        torch.save({'model': model.state_dict(), 'mu': mu.numpy(), 'sig': sig.numpy()},
                   phase1_ckpt)
        print(f'  Saved Phase 1 checkpoint: {phase1_ckpt}')

    # Phase 2: Unfreeze last 2 transformer blocks + mss_module
    print('\n' + '='*70)
    print('Phase 2: Partial fine-tune (last 2 blocks + MSS)')
    model.unfreeze_backbone_last_layers(n_blocks=2)
    n_trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Phase 2 trainable params: {n_trainable_p2:,}')
    train_phase(model, eeg_tr, tgt_tr_norm, epochs=150, lr=3e-5, bs=16, phase_name='Phase2')
    r2 = evaluate_e2e(model, eeg_te, tgt_te, mu, sig, 'Phase 2 (partial FT)')
    torch.save({'model': model.state_dict(), 'mu': mu.numpy(), 'sig': sig.numpy()},
               f'{SAVE_DIR}/fm_v6_phase2.pth')

    # Phase 3: Full fine-tune (very small LR)
    print('\n' + '='*70)
    print('Phase 3: Full fine-tune all params')
    model.unfreeze_all()
    n_trainable_p3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Phase 3 trainable params: {n_trainable_p3:,}')
    train_phase(model, eeg_tr, tgt_tr_norm, epochs=100, lr=5e-6, bs=16, phase_name='Phase3')
    r3 = evaluate_e2e(model, eeg_te, tgt_te, mu, sig, 'Phase 3 (full FT)')

    # Save checkpoint
    torch.save({
        'model': model.state_dict(),
        'mu': mu.numpy(), 'sig': sig.numpy(),
    }, f'{SAVE_DIR}/fm_v6_e2e.pth')

    # Summary
    print('\n' + '='*70)
    print('Summary:')
    print(f'  v3b (frozen BB):       Avg.R = 0.468')
    print(f'  v6 Phase 1 (frozen):   Avg.R = {r1["avg_r"]:.3f}')
    print(f'  v6 Phase 2 (partial):  Avg.R = {r2["avg_r"]:.3f}')
    print(f'  v6 Phase 3 (full):     Avg.R = {r3["avg_r"]:.3f}')
    print(f'  Target:                Avg.R = 0.600')
    print('='*70)

    with open(OUT_JSON, 'w') as f:
        json.dump({'phase1': r1, 'phase2': r2, 'phase3': r3}, f, indent=2)
    print(f'Saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
