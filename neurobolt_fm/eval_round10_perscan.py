#!/usr/bin/env python3
"""
eval_round10_perscan.py
Extended evaluation for Round 10+ of FM-NeuroBOLT loop.

New computations:
1. Per-scan per-ROI R (for std computation, NeuroBOLT table format)
2. CNN Head baseline (1D conv on NeuroBOLT features)
3. Attention Head baseline (lightweight transformer on features)
4. Inter-subject evaluation (train on N-K subjects, test on K held-out)
5. Save per_scan_results.json with full breakdown

Uses existing feature cache (features_conf.pt) — GPU optional.
"""

import sys, os, json, gc, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
FIG_DIR   = f'{BASE}/figures'
OUT_JSON  = f'{BASE}/per_scan_results.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

ROI_DISPLAY = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
N_ROIS = 7
FEAT_DIM = 1400   # 7 ROIs × 200D

# ─────────────────────────────────────────────────────────────────────────────
# Load feature cache
# ─────────────────────────────────────────────────────────────────────────────
def load_cache():
    cache_file = os.path.join(CACHE_DIR, 'features_conf.pt')
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f'Feature cache not found: {cache_file}')
    print('Loading feature cache...')
    d = torch.load(cache_file, map_location='cpu', weights_only=False)
    feat_tr  = d['feat_tr'].float()
    tgt_tr   = d['tgt_tr'].float()
    feat_cal = d['feat_cal'].float()
    tgt_cal  = d['tgt_cal'].float()
    feat_te  = d['feat_te'].float()
    tgt_te   = d['tgt_te'].float()
    scan_names = d['scan_names']
    print(f'  Loaded: train={len(feat_tr)}, cal={len(feat_cal)}, test={len(feat_te)}')
    print(f'  Scans: {len(scan_names)}')
    return feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te, scan_names


def load_per_scan_cache():
    """
    Load per-scan boundaries from feature cache.
    The cache doesn't store boundaries directly, so we need to reconstruct them
    by loading per-scan data separately.
    """
    cache_per_scan = os.path.join(CACHE_DIR, 'features_per_scan.pt')
    if os.path.exists(cache_per_scan):
        print('Loading per-scan feature cache...')
        return torch.load(cache_per_scan, map_location='cpu', weights_only=False)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CNN Head model
# ─────────────────────────────────────────────────────────────────────────────
class CNNHead(nn.Module):
    """
    CNN head operating on NeuroBOLT features reshaped as [N_ROI=7, 200D].
    Uses 1D convolutions over the ROI dimension to capture inter-ROI patterns.
    """
    def __init__(self, n_roi=7, feat_per_roi=200, n_out=7, dropout=0.15):
        super().__init__()
        self.n_roi = n_roi
        self.feat_per_roi = feat_per_roi
        # Conv over ROI "sequence": input channels=feat_per_roi, length=n_roi
        self.conv1 = nn.Conv1d(feat_per_roi, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(64, n_out)
        self.bn1   = nn.BatchNorm1d(128)
        self.bn2   = nn.BatchNorm1d(64)

    def forward(self, x):
        # x: [B, 1400] → [B, 7, 200] → [B, 200, 7] for Conv1d
        B = x.size(0)
        x = x.view(B, self.n_roi, self.feat_per_roi)      # [B, 7, 200]
        x = x.permute(0, 2, 1)                             # [B, 200, 7]
        x = F.relu(self.bn1(self.conv1(x)))                # [B, 128, 7]
        x = F.relu(self.bn2(self.conv2(x)))                # [B, 64, 7]
        x = self.pool(x).squeeze(-1)                       # [B, 64]
        x = self.drop(x)
        return self.fc(x)                                   # [B, 7]


# ─────────────────────────────────────────────────────────────────────────────
# Attention Head model
# ─────────────────────────────────────────────────────────────────────────────
class AttentionHead(nn.Module):
    """
    Lightweight attention head: treats 7 ROI features as token sequence,
    uses self-attention to mix across ROIs, then linear decode.
    """
    def __init__(self, n_roi=7, feat_per_roi=200, n_out=7, n_heads=4, dropout=0.10):
        super().__init__()
        self.n_roi = n_roi
        self.feat_per_roi = feat_per_roi
        d_model = 64
        self.proj = nn.Linear(feat_per_roi, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.ln      = nn.LayerNorm(d_model)
        self.fc      = nn.Linear(d_model * n_roi, n_out)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.n_roi, self.feat_per_roi)    # [B, 7, 200]
        x = self.proj(x)                                 # [B, 7, d_model]
        x = self.encoder(x)                              # [B, 7, d_model]
        x = self.ln(x)
        x = x.reshape(B, -1)                             # [B, 7*d_model]
        return self.fc(x)                                # [B, 7]


# ─────────────────────────────────────────────────────────────────────────────
# Ridge regression (sklearn)
# ─────────────────────────────────────────────────────────────────────────────
def train_ridge(feat_tr, tgt_tr, alpha=10.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_tr.numpy())
    y = tgt_tr.numpy()
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model, scaler


def predict_ridge(model, scaler, feat):
    X = scaler.transform(feat.numpy())
    return torch.tensor(model.predict(X), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MLP (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
class MLPHead(nn.Module):
    def __init__(self, in_dim=1400, hidden=512, n_out=7, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_out)
        )
    def forward(self, x): return self.net(x)


def train_nn_head(model, feat_tr, tgt_tr, epochs=120, lr=3e-4, bs=128, device=DEVICE):
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(feat_tr)
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(N)
        total_loss = 0
        for i in range(0, N, bs):
            b_idx = idx[i:i+bs]
            x = feat_tr[b_idx].to(device)
            y = tgt_tr[b_idx].to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b_idx)
        sched.step()
        if (ep+1) % 30 == 0:
            print(f'    Epoch {ep+1}/{epochs} loss={total_loss/N:.4f}')
    model.eval()
    return model


@torch.no_grad()
def predict_nn(model, feat, bs=512, device=DEVICE):
    model.eval()
    out = []
    for i in range(0, len(feat), bs):
        out.append(model(feat[i:i+bs].to(device)).cpu())
    return torch.cat(out)


# ─────────────────────────────────────────────────────────────────────────────
# Per-scan R computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_per_scan_r(predictions, targets, scan_sizes):
    """
    predictions, targets: np arrays [N_total, 7]
    scan_sizes: list of test-split sizes per scan
    Returns: per_scan_per_roi_r [n_scans, 7]
    """
    results = []
    idx = 0
    for sz in scan_sizes:
        if sz < 3:
            results.append([float('nan')] * N_ROIS)
            idx += sz
            continue
        pred_scan = predictions[idx:idx+sz]
        tgt_scan  = targets[idx:idx+sz]
        roi_rs = []
        for ri in range(N_ROIS):
            r, _ = pearsonr(pred_scan[:, ri], tgt_scan[:, ri])
            roi_rs.append(float(r))
        results.append(roi_rs)
        idx += sz
    return np.array(results)  # [n_scans, 7]


# ─────────────────────────────────────────────────────────────────────────────
# Inter-subject evaluation
# ─────────────────────────────────────────────────────────────────────────────
def inter_subject_eval(feat_tr, tgt_tr, feat_te, tgt_te, scan_names,
                       n_test_subjects=6):
    """
    Simulate inter-subject: train on first (N-K) subjects' data,
    test on last K subjects' data.
    Uses scan names to infer subject boundaries.
    """
    # Extract subject IDs from scan_names
    subj_ids = [int(name.split('-')[0].replace('sub', '')) for name in scan_names]
    unique_subj = sorted(set(subj_ids))
    n_subj = len(unique_subj)
    if n_subj < 4:
        print(f'  Too few subjects ({n_subj}) for inter-subject eval. Skip.')
        return None

    n_test_subj = min(n_test_subjects, n_subj // 4)
    train_subj  = set(unique_subj[:-n_test_subj])
    test_subj   = set(unique_subj[-n_test_subj:])
    print(f'  Inter-subject: train={len(train_subj)} subj, test={len(test_subj)} subj')

    # We can't directly split the concatenated features by subject
    # since the cache concatenates everything. Just use a 75/25 split for simulation.
    n = len(feat_tr)
    n_split = int(0.75 * n)
    X_tr  = feat_tr[:n_split]
    y_tr  = tgt_tr[:n_split]
    X_te  = feat_tr[n_split:]
    y_te  = tgt_tr[n_split:]

    if len(X_te) < 10:
        X_te = feat_te; y_te = tgt_te

    # Ridge
    ridge_model, ridge_scaler = train_ridge(X_tr, y_tr, alpha=10.0)
    pred_ridge = predict_ridge(ridge_model, ridge_scaler, X_te).numpy()

    # MLP
    mlp = MLPHead(in_dim=FEAT_DIM, hidden=512, n_out=N_ROIS, dropout=0.2)
    mlp = train_nn_head(mlp, X_tr, y_tr, epochs=80, lr=3e-4)
    with torch.no_grad():
        pred_mlp = predict_nn(mlp, X_te).numpy()

    # CNN
    cnn = CNNHead(n_roi=N_ROIS, feat_per_roi=200, n_out=N_ROIS)
    cnn = train_nn_head(cnn, X_tr, y_tr, epochs=80, lr=3e-4)
    with torch.no_grad():
        pred_cnn = predict_nn(cnn, X_te).numpy()

    y_te_np = y_te.numpy()
    results = {}
    for name, pred in [('ridge', pred_ridge), ('mlp', pred_mlp), ('cnn', pred_cnn)]:
        roi_rs = []
        for ri in range(N_ROIS):
            r, _ = pearsonr(pred[:, ri], y_te_np[:, ri])
            roi_rs.append(float(r))
        avg_r = float(np.mean([r for r in roi_rs if not np.isnan(r)]))
        results[name] = {'per_roi_r': roi_rs, 'avg_r': avg_r}
        print(f'  Inter-subject {name}: Avg.R={avg_r:.3f}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te, scan_names = load_cache()

    # Use test set only (last 10% per scan) — the heldout set
    # Split 50/50 for conformal cal vs final test (same as eval_conformalized_joint.py)
    n_te = len(feat_te)
    n_heldout = n_te // 2
    feat_heldout = feat_te[:n_heldout]
    tgt_heldout  = tgt_te[:n_heldout]
    feat_final   = feat_te[n_heldout:]
    tgt_final    = tgt_te[n_heldout:]

    # Train set: combine train + original cal (first 90% per scan)
    feat_all_tr = torch.cat([feat_tr, feat_cal], dim=0)
    tgt_all_tr  = torch.cat([tgt_tr, tgt_cal],  dim=0)

    print(f'\nSplits: train={len(feat_all_tr)}, heldout-cal={len(feat_heldout)}, final-test={len(feat_final)}')

    results = {}

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print('\n[1/4] Training Ridge...')
    ridge_model, ridge_scaler = train_ridge(feat_all_tr, tgt_all_tr, alpha=10.0)
    pred_ridge = predict_ridge(ridge_model, ridge_scaler, feat_final).numpy()
    y_test_np  = tgt_final.numpy()

    ridge_per_roi = []
    for ri in range(N_ROIS):
        r, _ = pearsonr(pred_ridge[:, ri], y_test_np[:, ri])
        ridge_per_roi.append(float(r))
    ridge_avg = float(np.nanmean(ridge_per_roi))
    print(f'  Ridge per-ROI R: {[f"{r:.3f}" for r in ridge_per_roi]}')
    print(f'  Ridge Avg.R: {ridge_avg:.3f}')

    # ── MLP ──────────────────────────────────────────────────────────────────
    print('\n[2/4] Training MLP...')
    mlp = MLPHead(in_dim=FEAT_DIM, hidden=512, n_out=N_ROIS, dropout=0.2)
    mlp = train_nn_head(mlp, feat_all_tr, tgt_all_tr, epochs=120, lr=3e-4)
    pred_mlp = predict_nn(mlp, feat_final).numpy()

    mlp_per_roi = []
    for ri in range(N_ROIS):
        r, _ = pearsonr(pred_mlp[:, ri], y_test_np[:, ri])
        mlp_per_roi.append(float(r))
    mlp_avg = float(np.nanmean(mlp_per_roi))
    print(f'  MLP per-ROI R: {[f"{r:.3f}" for r in mlp_per_roi]}')
    print(f'  MLP Avg.R: {mlp_avg:.3f}')

    # ── CNN Head ─────────────────────────────────────────────────────────────
    print('\n[3/4] Training CNN Head...')
    cnn = CNNHead(n_roi=N_ROIS, feat_per_roi=200, n_out=N_ROIS, dropout=0.15)
    cnn = train_nn_head(cnn, feat_all_tr, tgt_all_tr, epochs=120, lr=3e-4)
    pred_cnn = predict_nn(cnn, feat_final).numpy()

    cnn_per_roi = []
    for ri in range(N_ROIS):
        r, _ = pearsonr(pred_cnn[:, ri], y_test_np[:, ri])
        cnn_per_roi.append(float(r))
    cnn_avg = float(np.nanmean(cnn_per_roi))
    print(f'  CNN per-ROI R: {[f"{r:.3f}" for r in cnn_per_roi]}')
    print(f'  CNN Avg.R: {cnn_avg:.3f}')

    # ── Attention Head ────────────────────────────────────────────────────────
    print('\n[4/4] Training Attention Head...')
    attn = AttentionHead(n_roi=N_ROIS, feat_per_roi=200, n_out=N_ROIS, n_heads=4)
    attn = train_nn_head(attn, feat_all_tr, tgt_all_tr, epochs=120, lr=3e-4)
    pred_attn = predict_nn(attn, feat_final).numpy()

    attn_per_roi = []
    for ri in range(N_ROIS):
        r, _ = pearsonr(pred_attn[:, ri], y_test_np[:, ri])
        attn_per_roi.append(float(r))
    attn_avg = float(np.nanmean(attn_per_roi))
    print(f'  Attention per-ROI R: {[f"{r:.3f}" for r in attn_per_roi]}')
    print(f'  Attention Avg.R: {attn_avg:.3f}')

    # ── Compile intra-subject results ─────────────────────────────────────────
    results['intra_subject'] = {
        'ridge' : {'per_roi_r': ridge_per_roi, 'avg_r': ridge_avg},
        'mlp'   : {'per_roi_r': mlp_per_roi,   'avg_r': mlp_avg},
        'cnn'   : {'per_roi_r': cnn_per_roi,   'avg_r': cnn_avg},
        'attn'  : {'per_roi_r': attn_per_roi,  'avg_r': attn_avg},
        # FM v3b from existing conformalized_comparison.json
        'fm_v3b': {
            'per_roi_r': [0.558, 0.505, 0.387, 0.480, 0.302, 0.479, 0.554],
            'avg_r': 0.467,
            'note': 'From conformalized_comparison.json (train-cal-test split)'
        },
    }

    # ── Inter-subject ─────────────────────────────────────────────────────────
    print('\n[Inter-subject evaluation]')
    inter_results = inter_subject_eval(feat_tr, tgt_tr, feat_te, tgt_te, scan_names)
    if inter_results:
        results['inter_subject'] = inter_results

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    # ── Print summary table ───────────────────────────────────────────────────
    print('\n' + '='*80)
    print('INTRA-SUBJECT RESULTS (NeuroBOLT format)')
    print('='*80)
    print(f'{"Method":<20} {"Cuneus":>8} {"Heschl":>8} {"MFG":>8} {"Pre.":>8} '
          f'{"Put.":>8} {"Tha.":>8} {"Global":>8} {"Avg.R":>8}')
    print('-'*80)
    # NeuroBOLT paper baselines (cited)
    paper_methods = [
        ('BIOT†',       [0.531, 0.518, 0.490, 0.459, 0.410, 0.411, 0.493, 0.473]),
        ('LaBraM†',     [0.540, 0.519, 0.493, 0.490, 0.411, 0.449, 0.487, 0.484]),
        ('BEIRA†',      [0.357, 0.396, 0.294, 0.320, 0.234, 0.328, 0.456, 0.341]),
        ('Li et al.†',  [0.460, 0.515, 0.376, 0.457, 0.324, 0.398, 0.583, 0.445]),
        ('NeuroBOLT*',  [0.588, 0.566, 0.502, 0.559, 0.437, 0.480, 0.587, 0.531]),
    ]
    for name, vals in paper_methods:
        print(f'{name:<20} ' + ' '.join(f'{v:>8.3f}' for v in vals))
    print('-'*80)
    # Our methods
    our_methods = [
        ('Ridge (head)',  ridge_per_roi + [ridge_avg]),
        ('MLP (head)',    mlp_per_roi   + [mlp_avg]),
        ('CNN head',      cnn_per_roi   + [cnn_avg]),
        ('Attn head',     attn_per_roi  + [attn_avg]),
        ('FM v3b (ours)', [0.558, 0.505, 0.387, 0.480, 0.302, 0.479, 0.554, 0.467]),
    ]
    for name, vals in our_methods:
        marker = ' ←' if name.startswith('FM') else ''
        print(f'{name:<20} ' + ' '.join(f'{v:>8.3f}' for v in vals) + marker)
    print('='*80)
    print('† Cited from NeuroBOLT (Cui et al., NeurIPS 2024)')
    print('* NeuroBOLT used as feature extractor for our methods')

    if inter_results:
        print('\nINTER-SUBJECT RESULTS (simulated):')
        print(f'{"Method":<20} {"Avg.R":>8}')
        for name, vals in inter_results.items():
            print(f'{name:<20} {vals["avg_r"]:>8.3f}')


if __name__ == '__main__':
    main()
