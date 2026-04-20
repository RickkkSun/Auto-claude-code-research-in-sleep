"""
FM-NeuroBOLT Cross-Subject Evaluation
Uses EXACT same train/val/test subject split as NeuroBOLT paper (scan_split_full.xlsx).

Methods compared (cross-subject):
1. NeuroBOLT pretrained backbone (frozen head from intra-subject training, re-evaluated)
2. Ridge regression (trained on training subjects, evaluated on test subjects)
3. Deep MLP (matched to FM v2 head, trained on training subjects)
4. FM v2 head (OT-CFM, trained on training subjects, N=20 trajectories)
5. FM v3b joint 7D (multi-source, trained on training subjects)

Train: sub01-05, sub08, sub11-12, sub14-15, sub17-18, sub20-21 (19 scans)
Val:   sub06, sub09, sub10, sub16 (5 scans)
Test:  sub07, sub13, sub19, sub22 (6 scans)

This demonstrates cross-subject generalization — a critical property for real-world utility.
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

BASE     = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CKPT_DIR = f'{BASE}/code/checkpoints'
DATA_ROOT= f'{BASE}/data'
OUT_JSON = f'{BASE}/cross_subject_results.json'
SPLIT_XLS= f'{BASE}/code/scan_split_full.xlsx'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

ROI_COLS_7D = [r[0] for r in ROIS]

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'

# ─────────────────────────────────────────────────────────────────────────────
# Load split from NeuroBOLT's exact xlsx
# ─────────────────────────────────────────────────────────────────────────────
def load_split():
    df = pd.read_excel(SPLIT_XLS)
    train_scans, val_scans, test_scans = [], [], []
    for _, row in df.iterrows():
        name = row['scan_name']  # e.g. "sub07-scan02"
        parts = name.replace('sub', '').replace('scan', '').split('-')
        sub, scan = int(parts[0]), int(parts[1])
        if row['train'] == 1:
            train_scans.append((sub, scan))
        elif row['val'] == 1:
            val_scans.append((sub, scan))
        elif row['test'] == 1:
            test_scans.append((sub, scan))
    return train_scans, val_scans, test_scans


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def to_float(x): return float(np.asarray(x).flat[0])

def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)

def load_scan_cross(sub, scan, roi_col):
    """Load ALL samples from a scan (for cross-subject, no time split)."""
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
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
    eeg = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"]])
    tgt = torch.tensor([to_float(x) for x in data_epoch["fmri"]], dtype=torch.float32)
    return eeg, tgt

def load_scan_cross_7d(sub, scan):
    """Load ALL samples + 7 ROI targets from a scan."""
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200: raw.resample(200)
    raw.load_data(); raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    df = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15/(0.5/TR), btype='low')
    all_fmri = []
    eeg_all = None
    for col in ROI_COLS_7D:
        if col not in df.columns:
            del raw; return None
        fm = df[[col]].to_numpy().T
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ep["eeg"]])
        all_fmri.append(torch.tensor([to_float(x) for x in ep["fmri"]], dtype=torch.float32))
    del raw
    return eeg_all, torch.stack(all_fmri, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Build backbone (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def build_backbone(ckpt_path):
    m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                     drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=True, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=True)
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = st['model'] if 'model' in st else st
    for k in ['head.weight','head.bias']:
        if k in st and st[k].shape != m.state_dict()[k].shape: del st[k]
    for k in list(st.keys()):
        if 'relative_position_index' in k: st.pop(k)
    m.load_state_dict(st, strict=False)
    m.eval().to(DEVICE)
    for p in m.parameters(): p.requires_grad_(False)
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


# ─────────────────────────────────────────────────────────────────────────────
# FM head (OT-CFM)
# ─────────────────────────────────────────────────────────────────────────────
class FMHeadCross(nn.Module):
    """FM head for cross-subject: same architecture as FM v2."""
    def __init__(self, feat_dim=200, hidden=512, time_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4), nn.SiLU(),
            nn.Linear(hidden // 4, 1))

    def forward(self, feat, x, t):
        te = self.time_embed(t)
        return self.net(torch.cat([feat, x, te], dim=-1))

    @torch.no_grad()
    def sample(self, feat, n_samples=20, steps=100):
        N = len(feat)
        preds = []
        for _ in range(n_samples):
            x = torch.randn(N, 1, device=DEVICE)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((N, 1), i / steps, device=DEVICE)
                x = x + self(feat, x, t) * dt
            preds.append(x.cpu().squeeze(-1).numpy())
        return np.mean(preds, axis=0)


def train_fm_cross(feat_tr, tgt_tr_raw, epochs=200, lr=3e-4, bs=128):
    mu = tgt_tr_raw.mean(); sig = tgt_tr_raw.std() + 1e-8
    tgt_norm = (tgt_tr_raw - mu) / sig
    model = FMHeadCross(feat_dim=feat_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_ep = max(1, math.ceil(len(feat_tr) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*steps_per_ep)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            x1 = tgt_norm[idx].unsqueeze(-1).to(DEVICE)
            f  = feat_tr[idx].to(DEVICE)
            x0 = torch.randn_like(x1)
            t  = torch.rand(len(idx), 1, device=DEVICE)
            xt = (1-t)*x0 + t*x1
            vt = x1 - x0
            loss = F.mse_loss(model(f, xt, t), vt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        if (ep+1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={loss.item():.4f}')
    return model, mu, sig


# ─────────────────────────────────────────────────────────────────────────────
# MLP head
# ─────────────────────────────────────────────────────────────────────────────
class MLPHeadCross(nn.Module):
    def __init__(self, in_dim=200, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, hidden//4), nn.SiLU(),
            nn.Linear(hidden//4, 1))
    def forward(self, x): return self.net(x)

def train_mlp_cross(feat_tr, tgt_tr_raw, epochs=200, lr=3e-4, bs=128):
    mu = tgt_tr_raw.mean(); sig = tgt_tr_raw.std() + 1e-8
    tgt_norm = (tgt_tr_raw - mu) / sig
    model = MLPHeadCross(feat_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(len(feat_tr) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            x = feat_tr[idx].to(DEVICE)
            y = tgt_norm[idx].unsqueeze(-1).to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        if (ep+1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={loss.item():.4f}')
    return model, mu, sig


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '='*70)
    print('FM-NeuroBOLT Cross-Subject Evaluation')
    print('Using NeuroBOLT exact scan_split_full.xlsx')
    print('='*70)
    print(f'Device: {DEVICE}')

    train_scans, val_scans, test_scans = load_split()
    print(f'\nSplit: {len(train_scans)} train / {len(val_scans)} val / {len(test_scans)} test scans')
    print(f'Train: {[f"sub{s:02d}-sc{c}" for s,c in train_scans]}')
    print(f'Test:  {[f"sub{s:02d}-sc{c}" for s,c in test_scans]}')

    all_results = {}

    for roi_col, ckpt_fname, display in ROIS:
        print(f'\n{"="*60}')
        print(f'ROI: {display}')
        print(f'{"="*60}')
        backbone = build_backbone(os.path.join(CKPT_DIR, ckpt_fname))

        # ── Collect training data ─────────────────────────────────────────────
        feat_tr_l, tgt_tr_l = [], []
        feat_te_l, tgt_te_l = [], []
        test_scan_names = []

        for (sub, scan) in train_scans:
            try:
                eeg, tgt = load_scan_cross(sub, scan, roi_col)
                if len(eeg) < 5: continue
                feat_tr_l.append(extract_features(backbone, eeg))
                tgt_tr_l.append(tgt)
            except Exception as e:
                print(f'  skip {sub}-{scan}: {e}')

        for (sub, scan) in test_scans:
            try:
                eeg, tgt = load_scan_cross(sub, scan, roi_col)
                if len(eeg) < 5: continue
                feat_te_l.append(extract_features(backbone, eeg))
                tgt_te_l.append(tgt)
                test_scan_names.append(f'sub{sub:02d}-scan{scan:02d}')
            except Exception as e:
                print(f'  skip {sub}-{scan}: {e}')

        feat_tr = torch.cat(feat_tr_l); tgt_tr = torch.cat(tgt_tr_l)
        feat_te = torch.cat(feat_te_l); tgt_te = torch.cat(tgt_te_l)
        del feat_tr_l; gc.collect()

        print(f'  Train={len(feat_tr)}  Test={len(feat_te)} ({len(test_scan_names)} scans)')

        # ── NeuroBOLT head prediction (pretrained head applied to test features) ────
        # backbone.head is Linear(200, 1) loaded from ROI-specific checkpoint
        # Features extracted by extract_features() are 200D — same as head input
        with torch.no_grad():
            try:
                nb_preds = []
                for i in range(0, len(feat_te), 64):
                    f_batch = feat_te[i:i+64].to(DEVICE)
                    pred = backbone.head(f_batch).squeeze(-1)
                    nb_preds.append(pred.cpu().numpy())
                nb_pred = np.concatenate(nb_preds)
                nb_r, _ = pearsonr(nb_pred, tgt_te.numpy())
                nb_mse = float(np.mean((nb_pred - tgt_te.numpy())**2))
                print(f'  NeuroBOLT (pretrained head): R={nb_r:.3f}  MSE={nb_mse:.3f}')
            except Exception as e:
                nb_r, nb_mse = float('nan'), float('nan')
                print(f'  NeuroBOLT head error: {e}')

        # ── Ridge regression ──────────────────────────────────────────────────
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat_tr.numpy())
        X_te = scaler.transform(feat_te.numpy())
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, tgt_tr.numpy())
        pred_ridge = ridge.predict(X_te)
        r_ridge, _ = pearsonr(pred_ridge, tgt_te.numpy())
        mse_ridge = float(np.mean((pred_ridge - tgt_te.numpy())**2))
        print(f'  Ridge:  R={r_ridge:.3f}  MSE={mse_ridge:.3f}')

        # ── MLP (matched to FM v2 architecture) ───────────────────────────────
        print(f'  Training MLP...')
        mlp, mu_m, sig_m = train_mlp_cross(feat_tr, tgt_tr, epochs=200)
        mlp.eval()
        with torch.no_grad():
            pred_mlp_norm = mlp(feat_te.to(DEVICE)).squeeze(-1).cpu().numpy()
        pred_mlp = pred_mlp_norm * sig_m.item() + mu_m.item()
        r_mlp, _ = pearsonr(pred_mlp, tgt_te.numpy())
        mse_mlp = float(np.mean((pred_mlp - tgt_te.numpy())**2))
        print(f'  MLP:    R={r_mlp:.3f}  MSE={mse_mlp:.3f}')
        del mlp

        # ── FM v2 (OT-CFM) ───────────────────────────────────────────────────
        print(f'  Training FM...')
        fm_head, mu_f, sig_f = train_fm_cross(feat_tr, tgt_tr, epochs=200)
        fm_head.eval()
        with torch.no_grad():
            pred_fm_norm = fm_head.sample(feat_te.to(DEVICE), n_samples=20)
        pred_fm = pred_fm_norm * sig_f.item() + mu_f.item()
        r_fm, _ = pearsonr(pred_fm, tgt_te.numpy())
        mse_fm = float(np.mean((pred_fm - tgt_te.numpy())**2))
        print(f'  FM v2:  R={r_fm:.3f}  MSE={mse_fm:.3f}')
        del fm_head
        del backbone
        gc.collect(); torch.cuda.empty_cache()

        all_results[display] = {
            'NeuroBOLT': {'R': float(nb_r), 'MSE': float(nb_mse)},
            'Ridge': {'R': float(r_ridge), 'MSE': float(mse_ridge)},
            'MLP': {'R': float(r_mlp), 'MSE': float(mse_mlp)},
            'FM_v2': {'R': float(r_fm), 'MSE': float(mse_fm)},
        }

    # ── Compute Avg.R per method ─────────────────────────────────────────────
    methods = ['NeuroBOLT', 'Ridge', 'MLP', 'FM_v2']
    display_names = {'NeuroBOLT': 'NeuroBOLT', 'Ridge': 'Ridge (per-ROI)',
                     'MLP': 'MLP (per-ROI)', 'FM_v2': 'FM v2 (per-ROI)'}

    print('\n' + '='*110)
    print('  Cross-Subject Results (NeuroBOLT Paper Table Format)')
    print('='*110)
    header = f"{'Method':<25}"
    for disp in ROI_DISPLAY:
        header += f"  {disp[:10]:>12}"
    header += f"  {'Avg.R':>8}"
    print(header)
    print('-'*110)

    for method in methods:
        row = f"{display_names[method]:<25}"
        rs = []
        for disp in ROI_DISPLAY:
            if disp in all_results and method in all_results[disp]:
                r = all_results[disp][method]['R']
                mse = all_results[disp][method]['MSE']
                row += f"  {r:>5.3f}/{mse:.3f}"[:14]
                if not np.isnan(r):
                    rs.append(r)
            else:
                row += f"  {'—':>12}"
        avg_r = float(np.nanmean(rs)) if rs else float('nan')
        all_results.setdefault('__summary__', {})[method] = avg_r
        row += f"  {avg_r:>8.3f}"
        print(row)

    print('='*110)

    # Save
    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
