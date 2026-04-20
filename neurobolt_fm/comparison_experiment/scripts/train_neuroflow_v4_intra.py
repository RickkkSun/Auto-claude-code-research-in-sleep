"""
train_neuroflow_v4_intra.py — NeuroFlow v4 intra-subject evaluation.

Intra-subject protocol:
  For each scan independently:
    - Extract 1400D features (7 × 200D NeuroBOLT backbones)
    - Per-scan 80/10/10 split
    - Train Residual FM head (linear mean + FM on residuals + tau)
    - Evaluate: R, CRPS, FC-MAE

Aggregates results across all 29 scans and saves to neuroflow_v4_results.json.
"""

import sys, os, gc, json, math, warnings
BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, os.path.join(BASE, 'code'))
sys.path.insert(0, BASE)
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
ROIS = [
    ("Cuneus",                        "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",           "Heschl.pth",    "Heschl's"),
    ("Middle frontal gyrus anterior", "Midfront.pth",  "Mid.Front."),
    ("Precuneus anterior",            "Precuneus.pth", "Precuneus"),
    ("Putamen",                       "Putamen.pth",   "Putamen"),
    ("Thalamus",                       "Thalamus.pth",  "Thalamus"),
    ("global signal clean",           "glb.pth",       "Global"),
]
ROI_COLS    = [r[0] for r in ROIS]
ROI_DISPLAY = [r[2] for r in ROIS]
N_ROIS      = 7
FEAT_PER_BB = 200
FEAT_TOTAL  = 1400

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'

CKPT_DIR  = f'{BASE}/code/checkpoints'
DATA_ROOT = f'{BASE}/data'
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ─────────────────────────────────────────────────────────────────────────────
# NeuroBOLT backbone loading + feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_neurobolt(ckpt_path):
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
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def extract_neurobolt_features(backbone, eeg_tensor, bs=64):
    """Extract 200D features from one NeuroBOLT backbone."""
    from utils import get_input_chans
    ic  = get_input_chans(CH_NAMES)
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b, 'B N A T -> B N (A T)'), input_chans=None)
        out.append(backbone.head_act(xm + xt).cpu())
    return torch.cat(out)  # (N, 200)


def load_all_backbones():
    """Load all 7 ROI-specialized NeuroBOLT backbones."""
    print('\n[Loading 7 NeuroBOLT backbones]', flush=True)
    backbones = []
    for roi_col, ckpt_file, _ in ROIS:
        ckpt = os.path.join(CKPT_DIR, ckpt_file)
        bb   = build_neurobolt(ckpt)
        backbones.append(bb)
        print(f'  Loaded {ckpt_file}', flush=True)
    return backbones


def extract_1400d_features(backbones, eeg_tensor):
    """Extract 7 × 200D = 1400D features using all backbones."""
    parts = [extract_neurobolt_features(bb, eeg_tensor) for bb in backbones]
    return torch.cat(parts, dim=1)  # (N, 1400)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def to_float(x):
    return float(np.asarray(x).flat[0])


def load_scan(sub, scan):
    """Load EEG + all 7 ROI fMRI targets for one scan. Returns (eeg_tv, fmri_tv, eeg_te, fmri_te)."""
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    df    = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15 / (0.5 / TR), btype='low')

    all_fmri = []
    eeg_all  = None
    n = None

    for col in ROI_COLS:
        try:
            fm = df[[col]].to_numpy().T
        except KeyError:
            del raw
            return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            n = len(ep['eeg'])
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ep['eeg']])
        all_fmri.append(torch.tensor([to_float(x) for x in ep['fmri']], dtype=torch.float32))

    del raw
    fmri_all = torch.stack(all_fmri, dim=1)  # (n, 7)

    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / TR)

    return (eeg_all[:valcrop], fmri_all[:valcrop],
            eeg_all[valcrop:], fmri_all[valcrop:])


# ─────────────────────────────────────────────────────────────────────────────
# FM architecture (mirrors train_neuroflow_v4.py)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half-1, 1))
        emb   = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualFMHead(nn.Module):
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7, time_dim=32):
        super().__init__()
        self.n_rois   = n_rois
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
        f  = self.feat_proj(features)
        e  = self.time_emb(t)
        xi = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f, xi, e], dim=-1))

    @torch.no_grad()
    def sample(self, features, n_samples=50, num_steps=100):
        B  = features.shape[0]
        dt = 1.0 / num_steps
        out = []
        for _ in range(n_samples):
            x = torch.randn(B, self.n_rois, device=features.device)
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            out.append(x)
        return torch.stack(out)  # [K, B, 7]


def ot_cfm_loss(head, features, targets, sigma_min=0.001):
    B   = targets.shape[0]
    x0  = torch.randn_like(targets)
    x1  = targets
    t   = torch.rand(B, 1, device=targets.device)
    x_t = (1 - t) * x0 + t * x1 + sigma_min * torch.randn_like(x1)
    return F.mse_loss(head(features, x_t, t), x1 - x0)


def crps_score(samples, targets, exact=True):
    K, N, D = samples.shape
    e1 = (samples - targets.unsqueeze(0)).abs().mean()
    if exact:
        diff = (samples.unsqueeze(1) - samples.unsqueeze(0)).abs()
        e2   = 0.5 * diff.mean()
    else:
        perm = torch.randperm(K)
        e2   = 0.5 * (samples - samples[perm]).abs().mean()
    return (e1 - e2).item()


def corr_mat(x):
    x = x - x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp(min=1e-8)
    xn = x / std
    return (xn.T @ xn) / (len(xn) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-scan training
# ─────────────────────────────────────────────────────────────────────────────

def train_scan(feat_tv, tgt_tv, feat_te, tgt_te, epochs=200, lr=3e-4, bs=64):
    """Train Residual FM for one scan. Returns result dict."""
    N     = len(feat_tv)
    split = int(0.8 / 0.9 * N)   # 80% of trainval = train, 10% = val
    feat_tr, tgt_tr   = feat_tv[:split], tgt_tv[:split]
    feat_val, tgt_val = feat_tv[split:], tgt_tv[split:]

    if len(feat_tr) < 5 or len(feat_val) < 5:
        return None

    # Phase 1: Linear mean predictor
    lin = nn.Linear(FEAT_TOTAL, N_ROIS).to(DEVICE)
    opt_lin = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(50):
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx  = perm[i:i+bs]
            loss = F.mse_loss(lin(feat_tr[idx].to(DEVICE)), tgt_tr[idx].to(DEVICE))
            opt_lin.zero_grad(); loss.backward(); opt_lin.step()
    lin.eval()

    with torch.no_grad():
        mu_tr  = torch.cat([lin(feat_tr[i:i+256].to(DEVICE)).cpu()
                             for i in range(0, len(feat_tr), 256)])
        mu_val = torch.cat([lin(feat_val[i:i+256].to(DEVICE)).cpu()
                             for i in range(0, len(feat_val), 256)])
    res_tr  = tgt_tr  - mu_tr
    res_val = tgt_val - mu_val

    # Normalize residuals
    sig = res_tr.std(0).clamp(min=1e-8)
    res_tr_n  = res_tr  / sig
    res_val_n = res_val / sig

    # Phase 2: FM on residuals
    head = ResidualFMHead(feat_dim=FEAT_TOTAL).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, math.ceil(len(feat_tr) / bs)))

    best_crps, best_state, no_imp = 1e9, None, 0
    patience = 15

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            f   = feat_tr[idx].to(DEVICE)
            y   = res_tr_n[idx].to(DEVICE)
            loss = ot_cfm_loss(head, f, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (ep + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                sv = head.sample(feat_val.to(DEVICE), n_samples=10, num_steps=50)
                sv = (sv * sig.to(DEVICE)).cpu()
            vc = crps_score(sv, res_val, exact=False)
            if vc < best_crps:
                best_crps = vc
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()

    # Tau optimization on val — same formula as pooled script: mu + tau*(raw - mu)
    # Extended range [0.05, 1.5] allows shrinkage (tau<1) when FM is over-dispersed
    with torch.no_grad():
        sv = head.sample(feat_val.to(DEVICE), n_samples=50, num_steps=100)
        sv = (sv * sig.to(DEVICE)).cpu()
    best_tau_crps, best_tau = 1e9, 1.0
    mu_sv = sv.mean(0)   # [N_val, 7], FM sample mean on val residuals
    for tau in torch.linspace(0.05, 1.5, 60):
        # Interpolate/extrapolate around FM sample mean
        sc = mu_sv.unsqueeze(0) + tau * (sv - mu_sv.unsqueeze(0))
        c  = crps_score(sc, res_val, exact=False)
        if c < best_tau_crps:
            best_tau_crps = c
            best_tau = float(tau)

    # Test evaluation — same formula as val
    with torch.no_grad():
        st = head.sample(feat_te.to(DEVICE), n_samples=100, num_steps=100)
        st = (st * sig.to(DEVICE)).cpu()
    with torch.no_grad():
        mu_te = torch.cat([lin(feat_te[i:i+256].to(DEVICE)).cpu()
                           for i in range(0, len(feat_te), 256)])
    mu_st = st.mean(0)   # [N_te, 7], FM sample mean on test residuals
    # Total = linear mean + tau * (residuals - mean(residuals)) + mean(residuals)
    st_scaled = mu_st.unsqueeze(0) + best_tau * (st - mu_st.unsqueeze(0))  # [K, N_te, 7]
    st_total  = mu_te.unsqueeze(0) + st_scaled                              # [K, N_te, 7]

    pred_mean = st_total.mean(0)
    roi_r = [float(pearsonr(pred_mean[:, j].numpy(), tgt_te[:, j].numpy())[0])
             for j in range(N_ROIS)]

    crps_te = crps_score(st_total, tgt_te, exact=True)
    fc_mae  = (corr_mat(pred_mean) - corr_mat(tgt_te)).abs().mean().item()

    return {
        'avg_r'  : float(np.nanmean(roi_r)),
        'roi_r'  : roi_r,
        'crps'   : crps_te,
        'fc_mae' : fc_mae,
        'tau'    : best_tau,
        'n_te'   : len(tgt_te),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    backbones = load_all_backbones()

    scan_results = []
    scan_names   = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'\n[{pat}]', flush=True)
        try:
            data = load_scan(sub, scan)
            if data is None:
                print('  SKIP (missing ROI)', flush=True)
                continue
            eeg_tv, fmri_tv, eeg_te, fmri_te = data
            if len(eeg_te) < 5:
                print('  SKIP (short test)', flush=True)
                continue

            # Extract 1400D features
            feat_tv = extract_1400d_features(backbones, eeg_tv)
            feat_te = extract_1400d_features(backbones, eeg_te)

            res = train_scan(feat_tv, fmri_tv, feat_te, fmri_te,
                             epochs=200, lr=3e-4, bs=64)
            if res is None:
                print('  SKIP (too short for training)', flush=True)
                continue

            scan_results.append(res)
            scan_names.append(pat)
            print(f'  R={res["avg_r"]:.4f}  CRPS={res["crps"]:.4f}  FC-MAE={res["fc_mae"]:.4f}  tau={res["tau"]:.3f}')

            # Free GPU cache
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f'  SKIP ({e})', flush=True)

    if not scan_results:
        print('No valid scan results!')
        sys.exit(1)

    # Aggregate
    all_r    = [r['avg_r']  for r in scan_results]
    all_crps = [r['crps']   for r in scan_results]
    all_fc   = [r['fc_mae'] for r in scan_results]
    roi_r_mat = np.array([r['roi_r'] for r in scan_results])
    S = len(scan_results)

    # Scan-level bootstrap CI
    rng = np.random.default_rng(42)
    crps_boot = [np.array(all_crps)[rng.integers(0, S, S)].mean() for _ in range(500)]
    r_boot    = [np.array(all_r)[rng.integers(0, S, S)].mean() for _ in range(500)]
    crps_ci = (float(np.quantile(crps_boot, 0.025)), float(np.quantile(crps_boot, 0.975)))
    r_ci    = (float(np.quantile(r_boot, 0.025)),    float(np.quantile(r_boot, 0.975)))

    summary = {
        'avg_r'     : float(np.mean(all_r)),
        'avg_r_std' : float(np.std(all_r)),
        'r_ci'      : r_ci,
        'roi_r'     : roi_r_mat.mean(0).tolist(),
        'roi_r_mean': roi_r_mat.mean(0).tolist(),
        'crps'      : float(np.mean(all_crps)),
        'crps_ci'   : crps_ci,
        'fc_mae'    : float(np.mean(all_fc)),
        'mode'      : 'intra',
        'per_scan'  : [{'scan': n, **r} for n, r in zip(scan_names, scan_results)],
    }

    print(f'\n=== INTRA SUMMARY ({S} scans) ===')
    print(f'avg_R={summary["avg_r"]:.4f}±{summary["avg_r_std"]:.4f}  '
          f'R_95CI=[{r_ci[0]:.4f},{r_ci[1]:.4f}]')
    print(f'CRPS={summary["crps"]:.4f} [{crps_ci[0]:.4f},{crps_ci[1]:.4f}]  '
          f'FC-MAE={summary["fc_mae"]:.4f}')
    print('\nROI breakdown:')
    for name, r in zip([r[2] for r in ROIS], roi_r_mat.mean(0)):
        print(f'  {name:<14}: R={r:.4f}')

    # Save — same key format as train_neuroflow_v4.py
    out_path = os.path.join(OUT_DIR, 'neuroflow_v4_results.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing['neuroflow_v4_intra_tau'] = summary
    existing['neuroflow_intra'] = {
        'avg_r'     : summary['avg_r'],
        'roi_r'     : summary['roi_r'],
        'roi_r_mean': summary['roi_r_mean'],
        'crps'      : summary['crps'],
        'fc_mae'    : summary['fc_mae'],
        'crps_ci'   : crps_ci,
        'r_ci'      : r_ci,
        'mode'      : 'intra',
    }
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults saved to {out_path}  (key: neuroflow_intra)')
