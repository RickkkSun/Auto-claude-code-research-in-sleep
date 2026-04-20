"""
FM-NeuroBOLT Baselines — Fair comparison for Round 2

Implements matched baselines using IDENTICAL data splits, normalization, and backbones:
1. Ridge regression (per-ROI, per-backbone) — linear head
2. Deep MLP (per-ROI, per-backbone) — nonlinear head, same depth as FM head
3. Joint 7D MLP (shared glb.pth backbone) — joint regression, matched to FM v3

All use the same:
- 90% training data per scan (same as FM v2/v3)
- Z-score target normalization
- One backbone per ROI (same checkpoints as FM v2)

Key question: does FM beat matched MLP? If yes → FM-specific value demonstrated.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from timm.models import create_model
import models.model
from dataset_maker import preproc

# ─────────────────────────────────────────────────────────────────────────────
ROIS = [
    ("Cuneus",                        "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",           "Heschl.pth",    "Heschl's Gyrus"),
    ("Middle frontal gyrus anterior", "Midfront.pth",  "Mid. Frontal"),
    ("Precuneus anterior",            "Precuneus.pth", "Precuneus Ant."),
    ("Putamen",                       "Putamen.pth",   "Putamen"),
    ("Thalamus",                       "Thalamus.pth",  "Thalamus"),
    ("global signal clean",           "glb.pth",       "Global Signal"),
]
ROI_DISPLAY = [r[2] for r in ROIS]

ROI_COLS_7D = [
    "Cuneus",
    "Heschl\u2019s gyrus",
    "Middle frontal gyrus anterior",
    "Precuneus anterior",
    "Putamen",
    "Thalamus",
    "global signal clean",
]

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

CKPT_DIR  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/code/checkpoints'
DATA_ROOT = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/data'
NB_JSON   = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/neurobolt_intra_results.json'
V2_JSON   = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v2_results.json'
OUT_JSON  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/baseline_results.json'


def to_float(x):
    return float(np.asarray(x).flat[0])

def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)

def build_backbone(ckpt_path):
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
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m

@torch.no_grad()
def extract_features(backbone, eeg_tensor, bs=64):
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b,'B N A T -> B N (A T)'), input_chans=None)
        out.append(backbone.head_act(xm + xt).cpu())
    return torch.cat(out)

def load_scan(sub, scan, roi_col):
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')
    raw  = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200: raw.resample(200)
    raw.load_data(); raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    df = pd.read_pickle(fm_path)
    fm = df[[roi_col]].to_numpy().T
    b, a = butter(5, 0.15/(0.5/TR), btype='low')
    fm = filtfilt(b, a, fm, axis=1)
    fm, _ = preproc.normalize_data(fm)
    data_epoch, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
    del raw
    n = len(data_epoch["eeg"])
    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / TR)
    eeg_all   = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"]])
    eeg_train = eeg_all[:valcrop]
    tgt_train = torch.tensor([to_float(x) for x in data_epoch["fmri"][:valcrop]], dtype=torch.float32)
    eeg_test  = eeg_all[valcrop:]
    tgt_test  = torch.tensor([to_float(x) for x in data_epoch["fmri"][valcrop:]], dtype=torch.float32)
    return eeg_train, tgt_train, eeg_test, tgt_test

def load_scan_7d(sub, scan):
    """Load EEG + all 7 ROI targets."""
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')
    raw  = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200: raw.resample(200)
    raw.load_data(); raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    df = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15/(0.5/TR), btype='low')
    all_fmri = []
    eeg_all = None
    n = None
    for col in ROI_COLS_7D:
        try:
            fm = df[[col]].to_numpy().T
        except KeyError:
            del raw
            return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            n = len(ep["eeg"])
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ep["eeg"]])
        all_fmri.append(torch.tensor([to_float(x) for x in ep["fmri"]], dtype=torch.float32))
    del raw
    fmri_all = torch.stack(all_fmri, dim=1)  # (n, 7)
    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / TR)
    return eeg_all[:valcrop], fmri_all[:valcrop], eeg_all[valcrop:], fmri_all[valcrop:]


# ─────────────────────────────────────────────────────────────────────────────
# MLP head (matched to FM v2 architecture, minus the FM-specific parts)
# ─────────────────────────────────────────────────────────────────────────────
class MLPHead(nn.Module):
    """Deep MLP regression head — matched depth/width to FM head but no ODE."""
    def __init__(self, in_dim, hidden_dim=512, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp(feat_tr, tgt_tr_raw, out_dim=1, hidden_dim=512, epochs=200, lr=3e-4, bs=128):
    """Train MLP regressor with z-score normalization — same recipe as FM v2."""
    mu  = tgt_tr_raw.mean(0)
    sig = tgt_tr_raw.std(0) + 1e-8
    tgt_tr = (tgt_tr_raw - mu) / sig

    model = MLPHead(feat_tr.shape[1], hidden_dim, out_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(len(feat_tr) / bs))
    sch   = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            x   = feat_tr[idx].to(DEVICE)
            y   = tgt_tr[idx].to(DEVICE)
            if out_dim == 1:
                y = y.unsqueeze(-1)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        if (ep + 1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={loss.item():.4f}')

    return model, mu, sig


# ─────────────────────────────────────────────────────────────────────────────
# Per-ROI baselines (Ridge + MLP, using ROI-specific backbone — same as FM v2)
# ─────────────────────────────────────────────────────────────────────────────
def run_per_roi_baselines():
    """Run Ridge + MLP baselines per ROI with matched backbones."""
    results_ridge = {}
    results_mlp   = {}

    for roi_col, ckpt_fname, display in ROIS:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'\n[ROI] {display}')
        backbone = build_backbone(ckpt_path)

        feat_tr_l, tgt_tr_l = [], []
        feat_te_l, tgt_te_l = [], []
        scan_names_te = []

        for (sub, scan) in ALL_SCANS:
            try:
                et, tt, ete, tte = load_scan(sub, scan, roi_col)
                if len(ete) < 5: continue
                feat_tr_l.append(extract_features(backbone, et))
                tgt_tr_l.append(tt)
                feat_te_l.append(extract_features(backbone, ete))
                tgt_te_l.append(tte)
                scan_names_te.append(f'sub{sub:02d}-scan{scan:02d}')
            except Exception as e:
                pass

        feat_tr = torch.cat(feat_tr_l); tgt_tr = torch.cat(tgt_tr_l)
        feat_te = torch.cat(feat_te_l); tgt_te = torch.cat(tgt_te_l)
        del feat_tr_l; del backbone; gc.collect(); torch.cuda.empty_cache()

        # Ridge regression (sklearn)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat_tr.numpy())
        X_te = scaler.transform(feat_te.numpy())
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, tgt_tr.numpy())
        pred_ridge = ridge.predict(X_te)
        r_ridge, _ = pearsonr(pred_ridge, tgt_te.numpy())
        mse_ridge = float(np.mean((pred_ridge - tgt_te.numpy())**2))
        print(f'  Ridge: R={r_ridge:.3f}  MSE={mse_ridge:.3f}')
        results_ridge[display] = {'R': float(r_ridge), 'MSE': float(mse_ridge)}

        # MLP regression (same normalization + architecture as FM v2)
        print(f'  Training MLP...')
        mlp, mu, sig = train_mlp(feat_tr, tgt_tr, out_dim=1)
        mlp.eval()
        with torch.no_grad():
            pred_norm = mlp(feat_te.to(DEVICE)).squeeze().cpu().numpy()
        pred_mlp = pred_norm * sig.item() + mu.item()
        r_mlp, _ = pearsonr(pred_mlp, tgt_te.numpy())
        mse_mlp = float(np.mean((pred_mlp - tgt_te.numpy())**2))
        print(f'  MLP:   R={r_mlp:.3f}  MSE={mse_mlp:.3f}')
        results_mlp[display] = {'R': float(r_mlp), 'MSE': float(mse_mlp)}

        del feat_tr, feat_te

    results_ridge['avg_r'] = float(np.nanmean([results_ridge[d]['R'] for d in ROI_DISPLAY]))
    results_mlp['avg_r']   = float(np.nanmean([results_mlp[d]['R']   for d in ROI_DISPLAY]))
    return results_ridge, results_mlp


# ─────────────────────────────────────────────────────────────────────────────
# Joint 7D MLP baseline (shared glb.pth backbone — matched to FM v3)
# ─────────────────────────────────────────────────────────────────────────────
def run_joint_mlp_baseline():
    """Joint 7D MLP with glb.pth backbone — direct matched baseline for FM v3."""
    print('\n[Joint 7D MLP baseline] Loading backbone glb.pth...')
    backbone = build_backbone(os.path.join(CKPT_DIR, 'glb.pth'))

    feat_tr_l, tgt_tr_l = [], []
    feat_te_l, tgt_te_l = [], []
    scan_names_te = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        try:
            result = load_scan_7d(sub, scan)
            if result is None: continue
            et, tt, ete, tte = result
            if len(ete) < 5: continue
            feat_tr_l.append(extract_features(backbone, et))
            tgt_tr_l.append(tt)
            feat_te_l.append(extract_features(backbone, ete))
            tgt_te_l.append(tte)
            scan_names_te.append(pat)
        except Exception as e:
            pass

    feat_tr = torch.cat(feat_tr_l); tgt_tr = torch.cat(tgt_tr_l)
    feat_te = torch.cat(feat_te_l); tgt_te = torch.cat(tgt_te_l)
    del backbone; gc.collect(); torch.cuda.empty_cache()

    print(f'  Train={len(feat_tr)}  Test={len(feat_te)}')
    print('  Training Joint 7D MLP...')
    mlp, mu, sig = train_mlp(feat_tr, tgt_tr, out_dim=7, hidden_dim=512, epochs=300)
    mlp.eval()

    with torch.no_grad():
        pred_norm = mlp(feat_te.to(DEVICE)).cpu().numpy()  # (M, 7)
    pred = pred_norm * sig.numpy()[None, :] + mu.numpy()[None, :]  # de-normalize
    true = tgt_te.numpy()

    joint_results = {}
    for j, disp in enumerate(ROI_DISPLAY):
        p_j = pred[:, j]; t_j = true[:, j]
        r_g, _ = pearsonr(p_j, t_j)
        mse_g  = float(np.mean((p_j - t_j)**2))
        joint_results[disp] = {'R': float(r_g), 'MSE': float(mse_g)}
        print(f'    {disp}: R={r_g:.3f}  MSE={mse_g:.3f}')

    joint_results['avg_r'] = float(np.nanmean([joint_results[d]['R'] for d in ROI_DISPLAY]))
    print(f'  => Joint MLP Avg.R = {joint_results["avg_r"]:.3f}')

    # Covariance analysis: predicted vs true cross-ROI covariance
    pred_cov = np.corrcoef(pred.T)  # (7, 7)
    true_cov = np.corrcoef(true.T)  # (7, 7)
    cov_error = float(np.mean(np.abs(pred_cov[~np.eye(7, dtype=bool)] - true_cov[~np.eye(7, dtype=bool)])))
    print(f'  Covariance MAE (off-diag): {cov_error:.4f}')
    joint_results['cov_mae'] = cov_error

    return joint_results


# ─────────────────────────────────────────────────────────────────────────────
# Joint 7D MLP baseline with 7-backbone features (matched to FM v3b)
# ─────────────────────────────────────────────────────────────────────────────
def run_joint_mlp_7backbone_baseline():
    """
    Joint 7D MLP with all 7 specialized backbones — the EXACT fair baseline for FM v3b.
    Uses same 1400D features as FM v3b, but replaces the FM ODE with direct MLP regression.
    """
    print('\n[Joint 7D MLP (7-backbone) baseline]')
    print('  Loading all 7 specialized backbones and extracting features...')

    all_feat_train = []
    all_feat_test  = []
    tgt_train_list = []
    tgt_test_list  = []
    scan_names     = []

    # Load EEG + 7 ROI targets (one pass over scans)
    eeg_train_scans = []
    eeg_test_scans  = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        try:
            result = load_scan_7d(sub, scan)
            if result is None: continue
            et, tt, ete, tte = result
            if len(ete) < 5: continue
            eeg_train_scans.append(et)
            eeg_test_scans.append(ete)
            tgt_train_list.append(tt)
            tgt_test_list.append(tte)
            scan_names.append(pat)
        except Exception as e:
            pass

    print(f'  Loaded {len(scan_names)} scans')

    # Extract features from each of 7 specialized backbones
    for roi_col, ckpt_fname, display in ROIS:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'  Backbone: {display}...', end=' ', flush=True)
        backbone = build_backbone(ckpt_path)

        feat_tr_parts = [extract_features(backbone, et) for et in eeg_train_scans]
        feat_te_parts = [extract_features(backbone, ete) for ete in eeg_test_scans]

        all_feat_train.append(torch.cat(feat_tr_parts, dim=0))
        all_feat_test.append(torch.cat(feat_te_parts, dim=0))

        del backbone; gc.collect(); torch.cuda.empty_cache()
        print('done', flush=True)

    del eeg_train_scans, eeg_test_scans; gc.collect()

    feat_tr = torch.cat(all_feat_train, dim=1)  # (N_train, 1400)
    feat_te = torch.cat(all_feat_test, dim=1)   # (N_test, 1400)
    tgt_tr  = torch.cat(tgt_train_list, dim=0)  # (N_train, 7)
    tgt_te  = torch.cat(tgt_test_list, dim=0)   # (N_test, 7)

    print(f'  Feature shape: {feat_tr.shape}, Target: {tgt_tr.shape}')
    print('  Training Joint 7D MLP (7-backbone, 300 epochs)...')
    mlp, mu, sig = train_mlp(feat_tr, tgt_tr, out_dim=7, hidden_dim=512, epochs=300)
    mlp.eval()

    with torch.no_grad():
        pred_norm = mlp(feat_te.to(DEVICE)).cpu().numpy()
    pred = pred_norm * sig.numpy()[None, :] + mu.numpy()[None, :]
    true = tgt_te.numpy()

    joint7b_results = {}
    for j, disp in enumerate(ROI_DISPLAY):
        r_g, _ = pearsonr(pred[:, j], true[:, j])
        mse_g  = float(np.mean((pred[:, j] - true[:, j])**2))
        joint7b_results[disp] = {'R': float(r_g), 'MSE': float(mse_g)}
        print(f'    {disp}: R={r_g:.3f}  MSE={mse_g:.3f}')

    joint7b_results['avg_r'] = float(np.nanmean([joint7b_results[d]['R'] for d in ROI_DISPLAY]))
    print(f'  => Joint MLP (7-bb) Avg.R = {joint7b_results["avg_r"]:.3f}')

    # Covariance MAE
    pred_cov = np.corrcoef(pred.T)
    true_cov = np.corrcoef(true.T)
    cov_mae = float(np.mean(np.abs(pred_cov[~np.eye(7, dtype=bool)] - true_cov[~np.eye(7, dtype=bool)])))
    print(f'  Covariance MAE: {cov_mae:.4f}')
    joint7b_results['cov_mae'] = cov_mae

    return joint7b_results


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────
NB_KEY_MAP = {
    'Cuneus': 'Cuneus', "Heschl's Gyrus": 'Heschl', 'Mid. Frontal': 'MidFrontal',
    'Precuneus Ant.': 'PrecuneusAnt', 'Putamen': 'Putamen',
    'Thalamus': 'Thalamus', 'Global Signal': 'GlobalSig'
}

def print_table(results_dict):
    with open(NB_JSON) as f: nb = json.load(f)
    with open(V2_JSON) as f: v2 = json.load(f)

    cw = 15
    div = '=' * (30 + cw * 7 + 10)
    print(); print(div)
    print('  NeuroBOLT Paper Table 1 + Baselines + FM v2 (Pearson R / MSE)')
    print(div)
    hdr = f"{'Method':<30}" + ''.join(f'{d:>{cw}}' for d in ROI_DISPLAY) + f"{'Avg.R':>10}"
    print(hdr); print('-' * len(hdr))

    def row(name, data):
        rv   = [data.get(d, {}).get('R', float('nan'))   for d in ROI_DISPLAY]
        mv   = [data.get(d, {}).get('MSE', float('nan')) for d in ROI_DISPLAY]
        avg  = float(np.nanmean(rv))
        line = f'{name:<30}'
        for r, m in zip(rv, mv):
            line += f'{f"{r:.3f}/{m:.3f}":>{cw}}'
        line += f'{avg:>10.3f}'
        print(line)

    nb_data = {NB_KEY_MAP[d]: {'R': nb[NB_KEY_MAP[d]]['R'], 'MSE': nb[NB_KEY_MAP[d]]['MSE']}
               for d in ROI_DISPLAY}
    # rekey to display
    nb_disp = {d: {'R': nb[NB_KEY_MAP[d]]['R'], 'MSE': nb[NB_KEY_MAP[d]]['MSE']} for d in ROI_DISPLAY}
    v2_disp = {d: {'R': v2.get(d, {}).get('R', float('nan')), 'MSE': v2.get(d, {}).get('MSE', float('nan'))} for d in ROI_DISPLAY}

    row('NeuroBOLT (pretrained)', nb_disp)
    row('Ridge (per-ROI backbone)', results_dict.get('ridge', {}))
    row('MLP (per-ROI backbone)', results_dict.get('mlp', {}))
    row('Joint MLP (glb backbone)', results_dict.get('joint_mlp', {}))
    row('Joint MLP (7-bb, matched)', results_dict.get('joint_mlp_7bb', {}))
    row('FM v2 (per-ROI, frozen)', v2_disp)
    print(div)
    print()
    print('Note: Ridge and MLP use same per-ROI backbone as FM v2 (fair comparison).')
    print('Joint MLP uses glb.pth backbone (matched to FM v3).')
    print('FM v2 uses OT-CFM with N=20 MC samples; MLP uses deterministic prediction.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f'\n{"="*60}')
    print(f'FM-NeuroBOLT Baselines (Ridge + MLP + Joint MLP)')
    print(f'{"="*60}')
    print(f'Device: {DEVICE}')

    print('\n[Phase 1] Per-ROI baselines (Ridge + MLP)...')
    results_ridge, results_mlp = run_per_roi_baselines()

    print('\n[Phase 2] Joint 7D MLP baseline...')
    results_joint_mlp = run_joint_mlp_baseline()

    print('\n[Phase 3] Joint 7D MLP (7-backbone, matched to FM v3b)...')
    results_joint_mlp_7bb = run_joint_mlp_7backbone_baseline()

    all_results = {
        'ridge':          results_ridge,
        'mlp':            results_mlp,
        'joint_mlp':      results_joint_mlp,
        'joint_mlp_7bb':  results_joint_mlp_7bb,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')

    print_table(all_results)


if __name__ == '__main__':
    main()
