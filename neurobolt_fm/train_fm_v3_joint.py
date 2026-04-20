"""
FM-NeuroBOLT v3 — Joint 7-ROI Conditional Flow Matching

Key innovation: One JOINT 7D OT-CFM models p(y₁,..,y₇ | EEG) simultaneously.
This captures cross-ROI covariance (brain regions co-activate), contrasted with
v2's 7 independent 1D FMs each using a different ROI-specific backbone.

Architecture:
  EEG → NeuroBOLT backbone (glb.pth init, PARTIALLY fine-tuned) → 200-dim features
  Joint 7D OT-CFM:
    v_θ(features, x_t ∈ R^7, t ∈ [0,1]) → velocity ∈ R^7
    Network: 200+7+32 → 512 → 512 → 256 → 7 (SiLU + LayerNorm)

Training:
  - Load ALL 7 ROI targets per sample simultaneously
  - Shared backbone (last 4 blocks + MSS module unfrozen)
  - Differential LR: backbone=1e-5, head=3e-4
  - Z-score normalize each ROI independently, then train jointly
  - 300 epochs, batch=64, OneCycleLR, AdamW

Inference:
  - N=20 independent ODE trajectories (Euler, 100 steps) → mean per ROI
  - De-normalize per ROI using stored mu/sig
"""

import sys, os, gc, json, math, warnings
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROI_COLS = [
    "Cuneus",
    "Heschl\u2019s gyrus",
    "Middle frontal gyrus anterior",
    "Precuneus anterior",
    "Putamen",
    "Thalamus",
    "global signal clean",
]
ROI_DISPLAY = [
    "Cuneus", "Heschl's Gyrus", "Mid. Frontal",
    "Precuneus Ant.", "Putamen", "Thalamus", "Global Signal",
]
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

# Use global-signal checkpoint as backbone init (captures overall brain state)
BACKBONE_CKPT = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/code/checkpoints/glb.pth'
DATA_ROOT     = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/data'
SAVE_DIR      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/checkpoints_fm_v3'
NB_JSON       = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/neurobolt_intra_results.json'
V1_JSON       = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_results.json'
V2_JSON       = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v2_results.json'
OUT_JSON      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v3_results.json'
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone: partial fine-tuning (last 4 of 12 transformer blocks + MSS module)
# ─────────────────────────────────────────────────────────────────────────────
def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)

def build_backbone_finetune(ckpt_path, unfreeze_last_n=4):
    """Load backbone and unfreeze last N transformer blocks + MSS module."""
    m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                     drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=True, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=True)
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = st['model'] if 'model' in st else st
    for k in ['head.weight','head.bias']:
        if k in st and st[k].shape != m.state_dict()[k].shape:
            del st[k]
    for k in list(st.keys()):
        if 'relative_position_index' in k:
            st.pop(k)
    m.load_state_dict(st, strict=False)

    # Freeze all params first
    for p in m.parameters():
        p.requires_grad_(False)

    # Unfreeze last N transformer blocks (float params only)
    num_blocks = len(m.blocks)
    print(f'  Backbone: {num_blocks} blocks total, unfreezing last {unfreeze_last_n}')
    for i in range(num_blocks - unfreeze_last_n, num_blocks):
        for p in m.blocks[i].parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

    # Unfreeze MSS module (multi-scale spectral, float params only)
    for p in m.mss_module.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    # Unfreeze fc_norm (final layer norm)
    if m.fc_norm is not None:
        for p in m.fc_norm.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f'  Backbone trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    return m.to(DEVICE)


def extract_features_grad(backbone, eeg_tensor, bs=64):
    """Extract features WITH gradient tracking (for fine-tuning)."""
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b,'B N A T -> B N (A T)'), input_chans=None)
        feat = backbone.head_act(xm + xt)
        out.append(feat)
    return torch.cat(out, dim=0)  # (N, 200)


@torch.no_grad()
def extract_features_frozen(backbone, eeg_tensor, bs=64):
    """Extract features without gradient (for evaluation)."""
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b,'B N A T -> B N (A T)'), input_chans=None)
        feat = backbone.head_act(xm + xt)
        out.append(feat.cpu())
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Joint 7D FM Head with cross-ROI interaction layer
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class JointFMHead7D(nn.Module):
    """
    Joint 7D Conditional Flow Matching head.

    Models the JOINT velocity field for all 7 ROIs simultaneously,
    enabling the network to learn cross-ROI covariance structure.

    Input:  features (200) + x_t (7) + t_emb (32) = 239-dim
    Output: velocity ∈ R^7
    """
    def __init__(self, feature_dim=200, hidden_dim=512, n_rois=7, time_dim=32):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = SinusoidalEmbedding(time_dim)

        in_dim = feature_dim + n_rois + time_dim

        # ROI cross-interaction: learned mixing before the main network
        self.roi_proj = nn.Linear(n_rois, n_rois)  # cross-ROI coupling

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_rois),
        )

    def forward(self, features, x_t, t):
        """
        features: (B, 200)
        x_t:      (B, 7) — current noisy state on OT-CFM path
        t:        (B, 1) — time in [0, 1]
        returns:  (B, 7) — velocity estimate
        """
        t_emb = self.time_emb(t)                     # (B, 32)
        x_t_mixed = self.roi_proj(x_t)               # cross-ROI coupling
        inp = torch.cat([features, x_t_mixed, t_emb], dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def sample(self, features, n_samples=20, num_steps=100):
        """Monte-Carlo mean: average N independent 7D ODE trajectories."""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x = torch.randn(B, self.n_rois, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x)
        return torch.stack(preds).mean(0)  # (B, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading: ALL 7 ROIs per scan simultaneously
# ─────────────────────────────────────────────────────────────────────────────
def to_float(x):
    return float(np.asarray(x).flat[0])

def _load_scan_multi_roi_impl(sub, scan, roi_series_precomputed=None):
    """Load EEG + all 7 ROI fMRI targets for one scan."""
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    raw  = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    df = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15/(0.5/TR), btype='low')

    # Epoch each ROI using the same raw (same events → same EEG epochs, aligned fMRI)
    all_fmri = []
    eeg_all  = None
    n        = None

    for roi_col in ROI_COLS:
        try:
            fm = df[[roi_col]].to_numpy().T
        except KeyError:
            del raw
            return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        data_epoch, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)

        if eeg_all is None:
            # Build EEG tensor only once (same for all ROIs)
            n       = len(data_epoch["eeg"])
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32)
                                   for x in data_epoch["eeg"]])

        roi_vals = torch.tensor([to_float(x) for x in data_epoch["fmri"]], dtype=torch.float32)
        all_fmri.append(roi_vals)

    del raw  # free after all ROIs are processed

    # Stack to (n, 7)
    fmri_all = torch.stack(all_fmri, dim=1)  # (n, 7)

    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / TR)

    eeg_train = eeg_all[:valcrop]
    tgt_train = fmri_all[:valcrop]   # (valcrop, 7)
    eeg_test  = eeg_all[valcrop:]
    tgt_test  = fmri_all[valcrop:]   # (M, 7)

    return eeg_train, tgt_train, eeg_test, tgt_test


# ─────────────────────────────────────────────────────────────────────────────
# Training: Joint 7D OT-CFM with partial backbone fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def train_joint_fm(eeg_train_list, tgt_train_list, backbone,
                   epochs=300, lr_head=3e-4, lr_backbone=5e-6, bs=64, n_samples=20,
                   _ic=None):
    """
    Train joint 7D FM with partial backbone fine-tuning.

    eeg_train_list: list of (N_i, 26, 3200) EEG tensors (raw, kept in CPU RAM)
    tgt_train_list: list of (N_i, 7) fMRI target tensors
    """
    # Combine all scans
    eeg_all  = torch.cat(eeg_train_list, dim=0)   # (N_total, 26, 3200)
    tgt_all  = torch.cat(tgt_train_list, dim=0)   # (N_total, 7)
    N = len(eeg_all)
    print(f'  Total training samples: {N}')

    # Z-score normalize each ROI independently
    mu  = tgt_all.mean(0)   # (7,)
    sig = tgt_all.std(0) + 1e-8  # (7,)
    tgt_norm = (tgt_all - mu) / sig  # (N, 7)

    head = JointFMHead7D(feature_dim=200, hidden_dim=512, n_rois=N_ROIS).to(DEVICE)

    # Differential learning rates
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())

    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': head_params,     'lr': lr_head,     'weight_decay': 1e-4},
    ])

    steps_per_epoch = max(1, math.ceil(N / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_backbone, lr_head],
        total_steps=epochs * steps_per_epoch
    )

    print(f'  Steps per epoch: {steps_per_epoch}, Total steps: {epochs * steps_per_epoch}')

    ic = get_input_chans()  # precompute once

    backbone.train()
    head.train()

    for ep in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            eeg_b = eeg_all[idx].to(DEVICE)               # (B, 26, 3200)
            x1    = tgt_norm[idx].to(DEVICE)               # (B, 7) normalized targets

            # Extract features WITH gradient (partial fine-tuning)
            eeg_b_4d = rearrange(eeg_b / 100., 'B N (A T) -> B N A T', T=200)
            xt_feat = backbone.forward_ts_features(eeg_b_4d, input_chans=ic)
            xm_feat = backbone.mss_module(
                rearrange(eeg_b_4d, 'B N A T -> B N (A T)'), input_chans=None)
            features = backbone.head_act(xm_feat + xt_feat)  # (B, 200)

            # OT-CFM: straight-line path from x0~N(0,I_7) to x1
            t   = torch.rand(len(eeg_b), 1, device=DEVICE)
            x0  = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            target_vel = x1 - x0  # OT velocity

            v_pred = head(features, x_t, t)
            loss   = F.mse_loss(v_pred, target_vel)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params + head_params, 1.0)
            opt.step()
            sch.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={epoch_loss/n_batches:.4f}')

    return head, mu.numpy(), sig.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_joint(backbone, head, eeg_test_list, tgt_test_list, scan_names,
                   mu, sig, n_samples=20):
    """Evaluate joint 7D model on all test scans."""
    backbone.eval()
    head.eval()

    results = {d: {'preds': [], 'trues': []} for d in ROI_DISPLAY}

    for (nm, eeg_te, tgt_te) in zip(scan_names, eeg_test_list, tgt_test_list):
        if len(eeg_te) < 5:
            continue

        # Extract features (no gradient)
        feat_te = extract_features_frozen(backbone, eeg_te)  # (M, 200)

        # 7D ODE sampling
        pred_norm = head.sample(feat_te.to(DEVICE), n_samples=n_samples, num_steps=100)
        pred_norm = pred_norm.cpu().numpy()  # (M, 7)

        # De-normalize
        pred = pred_norm * sig[None, :] + mu[None, :]  # (M, 7)
        true = tgt_te.numpy()  # (M, 7)

        scan_rs = []
        for j, disp in enumerate(ROI_DISPLAY):
            p_j = pred[:, j]; t_j = true[:, j]
            if len(p_j) >= 5:
                r, _ = pearsonr(p_j, t_j)
                results[disp]['preds'].append(p_j)
                results[disp]['trues'].append(t_j)
                scan_rs.append(r)
        r_mean = np.mean(scan_rs) if scan_rs else float('nan')
        print(f'    {nm}: mean_R={r_mean:.3f}')

    # Global R and MSE per ROI
    roi_results = {}
    for j, disp in enumerate(ROI_DISPLAY):
        if not results[disp]['preds']:
            roi_results[disp] = {'R': float('nan'), 'MSE': float('nan')}
            continue
        all_p = np.concatenate(results[disp]['preds'])
        all_t = np.concatenate(results[disp]['trues'])
        r_g, _  = pearsonr(all_p, all_t)
        mse_g   = float(np.mean((all_p - all_t)**2))
        roi_results[disp] = {'R': float(r_g), 'MSE': float(mse_g)}
        print(f'  {disp}: Global R={r_g:.3f}  MSE={mse_g:.3f}')

    avg_r = float(np.nanmean([roi_results[d]['R'] for d in ROI_DISPLAY]))
    print(f'\n  => Avg.R = {avg_r:.3f}  (NeuroBOLT target: 0.406)')
    return roi_results, avg_r


# ─────────────────────────────────────────────────────────────────────────────
# Table printing (NeuroBOLT Paper format)
# ─────────────────────────────────────────────────────────────────────────────
NB_KEY_MAP = {
    'Cuneus': 'Cuneus', "Heschl's Gyrus": 'Heschl', 'Mid. Frontal': 'MidFrontal',
    'Precuneus Ant.': 'PrecuneusAnt', 'Putamen': 'Putamen',
    'Thalamus': 'Thalamus', 'Global Signal': 'GlobalSig'
}

def print_table(v3_results, avg_r_v3):
    with open(NB_JSON) as f: nb = json.load(f)

    # Try to load v1 and v2 results if available
    v1_results = v2_results = None
    if os.path.exists(V1_JSON):
        with open(V1_JSON) as f: v1_results = json.load(f)
    if os.path.exists(V2_JSON):
        with open(V2_JSON) as f: v2_results = json.load(f)

    cw = 16
    div = '=' * (32 + cw * N_ROIS + 10)
    print(); print(div)
    print('  NeuroBOLT Paper Table 1 Reproduction + Flow Matching Comparison (Pearson R / MSE)')
    print(div)
    hdr = f"{'Method':<32}" + ''.join(f'{d:>{cw}}' for d in ROI_DISPLAY) + f"{'Avg.R':>10}"
    print(hdr)
    print('-' * len(hdr))

    def print_row(name, r_vals, mse_vals):
        avg = float(np.nanmean(r_vals))
        line = f'{name:<32}'
        for r, m in zip(r_vals, mse_vals):
            line += f'{f"{r:.3f}/{m:.3f}":>{cw}}'
        line += f'{avg:>10.3f}'
        print(line)

    # NeuroBOLT baseline
    nb_r   = [nb[NB_KEY_MAP[d]]['R']   for d in ROI_DISPLAY]
    nb_mse = [nb[NB_KEY_MAP[d]]['MSE'] for d in ROI_DISPLAY]
    print_row('NeuroBOLT (pretrained)', nb_r, nb_mse)

    # FM v1 (if available)
    if v1_results:
        v1r   = [v1_results.get(d, {}).get('R', float('nan'))   for d in ROI_DISPLAY]
        v1mse = [v1_results.get(d, {}).get('MSE', float('nan')) for d in ROI_DISPLAY]
        print_row('FM v1 (frozen 70%)', v1r, v1mse)

    # FM v2 (if available)
    if v2_results:
        v2r   = [v2_results.get(d, {}).get('R', float('nan'))   for d in ROI_DISPLAY]
        v2mse = [v2_results.get(d, {}).get('MSE', float('nan')) for d in ROI_DISPLAY]
        print_row('FM v2 (frozen all%)', v2r, v2mse)

    # FM v3 (current)
    v3r   = [v3_results.get(d, {}).get('R', float('nan'))   for d in ROI_DISPLAY]
    v3mse = [v3_results.get(d, {}).get('MSE', float('nan')) for d in ROI_DISPLAY]
    print_row('FM v3 (Joint 7D, finetune)', v3r, v3mse)

    print(div)
    print('\nNote: R/MSE per cell.  Avg.R = mean Pearson R across 7 ROIs.')
    print('FM v3: Joint 7D OT-CFM, shared backbone (glb.pth init),')
    print('       last 4 blocks + MSS module fine-tuned (lr=5e-6),')
    print('       head lr=3e-4, 300 epochs, N=20 MC samples.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f'\n{"="*60}')
    print(f'FM-NeuroBOLT v3 — Joint 7D Flow Matching')
    print(f'{"="*60}')
    print(f'Device: {DEVICE}')
    print(f'Backbone init: {BACKBONE_CKPT}')
    print(f'Architecture: Joint 7D OT-CFM + partial backbone fine-tuning')
    print()

    # Build backbone (partially fine-tunable)
    print('[1/3] Building backbone with partial fine-tuning...')
    backbone = build_backbone_finetune(BACKBONE_CKPT, unfreeze_last_n=4)

    # Load all scan data
    print('\n[2/3] Loading all scan data (EEG + 7 ROI targets)...')
    eeg_train_list  = []
    tgt_train_list  = []
    eeg_test_list   = []
    tgt_test_list   = []
    scan_names_te   = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'  {pat}...', end=' ', flush=True)
        try:
            result = _load_scan_multi_roi_impl(sub, scan)
            if result is None:
                print('SKIP (missing ROI column)')
                continue
            et, tt, ete, tte = result
            if len(ete) < 5:
                print('SKIP (short test)')
                continue
            eeg_train_list.append(et)
            tgt_train_list.append(tt)
            eeg_test_list.append(ete)
            tgt_test_list.append(tte)
            scan_names_te.append(pat)
            print(f'tr={len(et)} te={len(ete)}')
        except Exception as e:
            print(f'SKIP ({e})')

    print(f'\n  Loaded {len(scan_names_te)} scans')
    print(f'  Total train: {sum(len(x) for x in eeg_train_list)} samples')
    print(f'  Total test:  {sum(len(x) for x in eeg_test_list)} samples')

    # Train joint 7D FM
    print('\n[3/3] Training joint 7D FM with partial backbone fine-tuning...')
    head, mu, sig = train_joint_fm(
        eeg_train_list, tgt_train_list, backbone,
        epochs=300, lr_head=3e-4, lr_backbone=5e-6, bs=64, n_samples=20
    )

    # Evaluate
    print('\n[Evaluation]')
    roi_results, avg_r = evaluate_joint(
        backbone, head, eeg_test_list, tgt_test_list, scan_names_te,
        mu, sig, n_samples=20
    )

    # Save checkpoint (save full backbone state for reproducibility)
    save_path = os.path.join(SAVE_DIR, 'fm_v3_joint7d.pth')
    torch.save({
        'head': head.state_dict(),
        'backbone': backbone.state_dict(),
        'mu': mu.tolist(),
        'sig': sig.tolist(),
        'roi_display': ROI_DISPLAY,
    }, save_path)
    print(f'\nSaved -> {save_path}')

    # Save results
    v3_results_save = {d: {'R': roi_results[d]['R'], 'MSE': roi_results[d]['MSE']}
                       for d in ROI_DISPLAY}
    v3_results_save['avg_r'] = avg_r
    with open(OUT_JSON, 'w') as f:
        json.dump(v3_results_save, f, indent=2)
    print(f'Saved -> {OUT_JSON}')

    # Print paper table
    print_table(roi_results, avg_r)


if __name__ == '__main__':
    main()
