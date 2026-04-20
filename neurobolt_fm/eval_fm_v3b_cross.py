"""
FM-NeuroBOLT v3b Cross-Subject Evaluation

Trains a new joint 7D FM v3b on cross-subject training scans (18 scans)
and evaluates on held-out test scans (6 scans).
Updates cross_subject_results.json with FM_v3b entry.

Cross-subject split: uses NeuroBOLT's exact scan_split_full.xlsx.
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
ROI_COLS    = [r[0] for r in ROIS]
ROI_DISPLAY = [r[2] for r in ROIS]
N_ROIS = 7
FEAT_TOTAL = N_ROIS * 200  # 1400

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'


def to_float(x): return float(np.asarray(x).flat[0])

def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)

def load_split():
    df = pd.read_excel(SPLIT_XLS)
    train_scans, val_scans, test_scans = [], [], []
    for _, row in df.iterrows():
        name = row['scan_name']
        parts = name.replace('sub','').replace('scan','').split('-')
        sub, scan = int(parts[0]), int(parts[1])
        if row['train'] == 1:
            train_scans.append((sub, scan))
        elif row['val'] == 1:
            val_scans.append((sub, scan))
        elif row['test'] == 1:
            test_scans.append((sub, scan))
    return train_scans, val_scans, test_scans

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

def load_scan_7d(sub, scan):
    """Load all samples + all 7 ROI targets for one cross-subject scan."""
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
    for col in ROI_COLS:
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
    return eeg_all, torch.stack(all_fmri, dim=1)  # eeg (N, 26, 3200), tgt (N, 7)


# ─────────────────────────────────────────────────────────────────────────────
# FM v3b architecture (same as intra-subject)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half-1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class MultiSourceFMHead(nn.Module):
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7, time_dim=32):
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
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4), nn.SiLU(),
            nn.Linear(hidden_dim//4, n_rois),
        )

    def forward(self, features, x_t, t):
        f_proj = self.feat_proj(features)
        t_emb  = self.time_emb(t)
        x_t_i  = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f_proj, x_t_i, t_emb], dim=-1))

    @torch.no_grad()
    def sample(self, features, n_samples=20, num_steps=100):
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


def train_fm_v3b_cross(feat_tr, tgt_tr, epochs=200, lr=3e-4, bs=128):
    N = len(feat_tr)
    mu  = tgt_tr.mean(0); sig = tgt_tr.std(0) + 1e-8
    tgt_norm = (tgt_tr - mu) / sig

    head = MultiSourceFMHead(feat_dim=FEAT_TOTAL).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_ep = max(1, math.ceil(N / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        ep_loss = 0.0; nb = 0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            c  = feat_tr[idx].to(DEVICE)
            x1 = tgt_norm[idx].to(DEVICE)
            t  = torch.rand(len(c), 1, device=DEVICE)
            x0 = torch.randn_like(x1)
            xt = (1-t)*x0 + t*x1
            loss = F.mse_loss(head(c, xt, t), x1 - x0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sch.step()
        if (ep+1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={ep_loss/nb:.4f}')

    return head, mu.numpy(), sig.numpy()


def main():
    print('\n' + '='*70)
    print('FM-NeuroBOLT v3b Cross-Subject Evaluation')
    print('='*70)
    print(f'Device: {DEVICE}')

    train_scans, val_scans, test_scans = load_split()
    print(f'Split: {len(train_scans)} train / {len(val_scans)} val / {len(test_scans)} test')

    # ── Step 1: Load all scans into memory ───────────────────────────────────
    print('\n[1/3] Loading scan data...')
    eeg_tr_scans, tgt_tr_scans = [], []
    eeg_te_scans, tgt_te_scans = [], []
    test_scan_names = []

    for (sub, scan) in train_scans:
        try:
            result = load_scan_7d(sub, scan)
            if result is None: continue
            eeg, tgt = result
            if len(eeg) < 5: continue
            eeg_tr_scans.append(eeg); tgt_tr_scans.append(tgt)
            print(f'  train sub{sub:02d}-scan{scan:02d}: {len(eeg)} samples')
        except Exception as e:
            print(f'  skip train sub{sub:02d}-scan{scan:02d}: {e}')

    for (sub, scan) in test_scans:
        try:
            result = load_scan_7d(sub, scan)
            if result is None: continue
            eeg, tgt = result
            if len(eeg) < 5: continue
            eeg_te_scans.append(eeg); tgt_te_scans.append(tgt)
            test_scan_names.append(f'sub{sub:02d}-scan{scan:02d}')
            print(f'  test  sub{sub:02d}-scan{scan:02d}: {len(eeg)} samples')
        except Exception as e:
            print(f'  skip test sub{sub:02d}-scan{scan:02d}: {e}')

    n_tr = sum(len(x) for x in eeg_tr_scans)
    n_te = sum(len(x) for x in eeg_te_scans)
    print(f'\n  Train: {n_tr} samples ({len(eeg_tr_scans)} scans)')
    print(f'  Test:  {n_te} samples ({len(eeg_te_scans)} scans)')

    # ── Step 2: Extract multi-source features ────────────────────────────────
    print('\n[2/3] Extracting multi-source features (7 backbones × scans)...')
    all_feat_tr = []  # one (200,) tensor per backbone per scan sample
    all_feat_te = []

    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'  Backbone {j+1}/7: {display}...')
        backbone = build_backbone(ckpt_path)

        feat_tr_parts = [extract_features(backbone, eeg) for eeg in eeg_tr_scans]
        feat_te_parts = [extract_features(backbone, eeg) for eeg in eeg_te_scans]

        all_feat_tr.append(torch.cat(feat_tr_parts, dim=0))
        all_feat_te.append(torch.cat(feat_te_parts, dim=0))

        del backbone; gc.collect(); torch.cuda.empty_cache()

    # Free EEG tensors
    del eeg_tr_scans, eeg_te_scans; gc.collect()

    feat_tr = torch.cat(all_feat_tr, dim=1)  # (N_tr, 1400)
    feat_te = torch.cat(all_feat_te, dim=1)  # (N_te, 1400)
    tgt_tr  = torch.cat(tgt_tr_scans, dim=0)  # (N_tr, 7)
    tgt_te  = torch.cat(tgt_te_scans, dim=0)  # (N_te, 7)

    print(f'  feat_tr: {feat_tr.shape}, feat_te: {feat_te.shape}')

    # Per-scan test slices
    test_slices = []
    idx = 0
    for tgt in tgt_te_scans:
        n = len(tgt)
        test_slices.append(slice(idx, idx+n)); idx += n

    # ── Step 3: Train FM v3b on training scans ───────────────────────────────
    print('\n[3/3] Training FM v3b on cross-subject training scans...')
    head, mu, sig = train_fm_v3b_cross(feat_tr, tgt_tr, epochs=200)

    # ── Step 4: Evaluate on test scans ───────────────────────────────────────
    print('\n[4/4] Evaluating on cross-subject test scans...')
    head.eval()
    with torch.no_grad():
        pred_norm = head.sample(feat_te.to(DEVICE), n_samples=20, num_steps=100)
    pred_norm = pred_norm.cpu().numpy()
    pred = pred_norm * sig[None, :] + mu[None, :]
    true = tgt_te.numpy()

    # Per-ROI R values
    roi_rs = {}
    for j, disp in enumerate(ROI_DISPLAY):
        r, _ = pearsonr(pred[:, j], true[:, j])
        mse   = float(np.mean((pred[:, j] - true[:, j])**2))
        roi_rs[disp] = {'R': float(r), 'MSE': float(mse)}
        print(f'  {disp}: R={r:.3f}  MSE={mse:.3f}')

    avg_r = float(np.nanmean([v['R'] for v in roi_rs.values()]))
    print(f'\n  FM v3b Cross-Subject Avg.R = {avg_r:.3f}')

    # CovMAE on cross-subject test set
    pred_corr = np.corrcoef(pred.T)
    true_corr = np.corrcoef(true.T)
    cov_mae = float(np.mean(np.abs(pred_corr - true_corr)[~np.eye(N_ROIS, dtype=bool)]))
    print(f'  Cross-ROI CovMAE (cross-subject): {cov_mae:.4f}')

    # ── Update cross_subject_results.json ────────────────────────────────────
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for disp in ROI_DISPLAY:
        if disp not in all_results:
            all_results[disp] = {}
        all_results[disp]['FM_v3b'] = {
            'R': roi_rs[disp]['R'], 'MSE': roi_rs[disp]['MSE']
        }

    if '__summary__' not in all_results:
        all_results['__summary__'] = {}
    all_results['__summary__']['FM_v3b'] = avg_r

    all_results['__fm_v3b_cov_mae_cross__'] = cov_mae

    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nUpdated {OUT_JSON} with FM_v3b cross-subject results')

    # Summary table
    print('\n' + '='*90)
    print('Cross-Subject Results (updated)')
    print('='*90)
    methods = ['NeuroBOLT', 'Ridge', 'MLP', 'FM_v2', 'FM_v3b']
    header = f"{'Method':<22}"
    for disp in ROI_DISPLAY:
        header += f"  {disp[:8]:>8}"
    header += f"  {'Avg.R':>7}"
    print(header); print('-'*90)

    for method in methods:
        row = f"{method:<22}"
        rs = []
        for disp in ROI_DISPLAY:
            r = all_results.get(disp, {}).get(method, {}).get('R', float('nan'))
            row += f"  {r:>8.3f}"
            if not np.isnan(r): rs.append(r)
        avg = float(np.nanmean(rs)) if rs else float('nan')
        row += f"  {avg:>7.3f}"
        print(row)
    print('='*90)


if __name__ == '__main__':
    main()
