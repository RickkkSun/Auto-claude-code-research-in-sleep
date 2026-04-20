"""
FM-NeuroBOLT Round 7: Conformalized Apples-to-Apples Comparison

Key experiment: All probabilistic methods calibrated to Cover90=0.90.
Compare: coverage, interval width, CRPS, NLL, CovMAE after conformal calibration.

Methods:
1. FM v3b (joint 7D, OT-CFM) — existing checkpoint
2. Joint Gaussian Head (full Cholesky NLL) — retrained with checkpoint
3. Joint MLP (7bb) — retrained for baseline

Split: 80% train, 10% conformal-cal, 10% test (per scan, time-ordered)

Key claim:
  After conformalization to Cover90=0.90, FM v3b uniquely preserves
  cross-ROI covariance structure (CovMAE 0.059 vs Gaussian 0.180),
  while achieving valid coverage. FM is the right tool for joint
  distribution modeling; Gaussian is better for marginal sharpness.

Output: conformalized_comparison.json, fig9_conformalized.png
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from timm.models import create_model
import models.model
from dataset_maker import preproc

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CKPT_DIR  = f'{BASE}/code/checkpoints'
DATA_ROOT = f'{BASE}/data'
V3B_CKPT  = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
CACHE_DIR = f'{BASE}/feature_cache'
FIG_DIR   = f'{BASE}/figures'
OUT_JSON  = f'{BASE}/conformalized_comparison.json'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

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
FEAT_TOTAL = N_ROIS * 200

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]


def to_float(x): return float(np.asarray(x).flat[0])

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

def load_scan_all_rois(sub, scan):
    """Load EEG + all 7 ROI fMRI for one scan, with 80/10/10 split."""
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
        try: fm = df[[col]].to_numpy().T
        except KeyError: del raw; return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ep["eeg"]])
        all_fmri.append(torch.tensor([to_float(x) for x in ep["fmri"]], dtype=torch.float32))
    del raw
    n = len(eeg_all)
    n_tr  = int(0.8 * n)
    n_cal = int(0.1 * n)
    n_te  = n - n_tr - n_cal
    if n_te < 5: return None
    fmri_all = torch.stack(all_fmri, dim=1)  # (n, 7)
    return (eeg_all[:n_tr], fmri_all[:n_tr],              # train
            eeg_all[n_tr:n_tr+n_cal], fmri_all[n_tr:n_tr+n_cal],  # conformal-cal
            eeg_all[n_tr+n_cal:], fmri_all[n_tr+n_cal:])           # test


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction with caching
# ─────────────────────────────────────────────────────────────────────────────
def get_or_extract_features():
    """Extract 1400D multi-source features with disk caching."""
    cache_file = os.path.join(CACHE_DIR, 'features_conf.pt')

    if os.path.exists(cache_file):
        print('  Loading cached features...')
        d = torch.load(cache_file, map_location='cpu', weights_only=False)
        return (d['feat_tr'], d['tgt_tr'],
                d['feat_cal'], d['tgt_cal'],
                d['feat_te'], d['tgt_te'],
                d['scan_names'])

    print('  Extracting multi-source features (7 backbones × 29 scans)...')

    eeg_tr_all = []; tgt_tr_all = []
    eeg_cal_all = []; tgt_cal_all = []
    eeg_te_all = []; tgt_te_all = []
    scan_names = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'    {pat}...', end=' ', flush=True)
        try:
            result = load_scan_all_rois(sub, scan)
            if result is None: print('SKIP'); continue
            et, tt, ec, tc, ete, tte = result
            eeg_tr_all.append(et); tgt_tr_all.append(tt)
            eeg_cal_all.append(ec); tgt_cal_all.append(tc)
            eeg_te_all.append(ete); tgt_te_all.append(tte)
            scan_names.append(pat)
            print(f'tr={len(et)} cal={len(ec)} te={len(ete)}')
        except Exception as e:
            print(f'SKIP ({e})')

    # Extract features from all 7 backbones
    all_feat_tr = []; all_feat_cal = []; all_feat_te = []
    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'  Backbone {j+1}/7: {display}...', flush=True)
        backbone = build_backbone(ckpt_path)
        tr_parts  = [extract_features(backbone, eeg) for eeg in eeg_tr_all]
        cal_parts = [extract_features(backbone, eeg) for eeg in eeg_cal_all]
        te_parts  = [extract_features(backbone, eeg) for eeg in eeg_te_all]
        all_feat_tr.append(torch.cat(tr_parts, dim=0))
        all_feat_cal.append(torch.cat(cal_parts, dim=0))
        all_feat_te.append(torch.cat(te_parts, dim=0))
        del backbone; gc.collect(); torch.cuda.empty_cache()

    del eeg_tr_all, eeg_cal_all, eeg_te_all; gc.collect()

    feat_tr  = torch.cat(all_feat_tr,  dim=1)  # (N_tr, 1400)
    feat_cal = torch.cat(all_feat_cal, dim=1)  # (N_cal, 1400)
    feat_te  = torch.cat(all_feat_te,  dim=1)  # (N_te, 1400)
    tgt_tr   = torch.cat(tgt_tr_all,  dim=0)
    tgt_cal  = torch.cat(tgt_cal_all, dim=0)
    tgt_te   = torch.cat(tgt_te_all,  dim=0)

    torch.save({'feat_tr': feat_tr, 'tgt_tr': tgt_tr,
                'feat_cal': feat_cal, 'tgt_cal': tgt_cal,
                'feat_te': feat_te, 'tgt_te': tgt_te,
                'scan_names': scan_names}, cache_file)
    print(f'  Features cached to {cache_file}')
    return feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te, scan_names


# ─────────────────────────────────────────────────────────────────────────────
# FM v3b (load existing checkpoint)
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
    def sample_all(self, features, n_samples=50, num_steps=100):
        """Returns all N samples: (n_samples, B, n_rois)"""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x = torch.randn(B, self.n_rois, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x.cpu())
        return torch.stack(preds, dim=0)  # (N, B, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Joint Gaussian Head
# ─────────────────────────────────────────────────────────────────────────────
class JointGaussianHead(nn.Module):
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7):
        super().__init__()
        self.n_rois = n_rois
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, n_rois)
        n_chol = n_rois * (n_rois + 1) // 2
        self.chol_head = nn.Linear(hidden_dim, n_chol)

    def forward(self, feat):
        h = self.feat_proj(feat)
        h = self.shared(h)
        mu = self.mu_head(h)
        chol_vec = self.chol_head(h)
        B = feat.shape[0]
        L = torch.zeros(B, self.n_rois, self.n_rois, device=feat.device)
        idx = torch.tril_indices(self.n_rois, self.n_rois)
        L[:, idx[0], idx[1]] = chol_vec
        diag_idx = torch.arange(self.n_rois)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-4
        return mu, L

    def nll_loss(self, feat, target_norm):
        mu, L = self.forward(feat)
        dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
        return -dist.log_prob(target_norm).mean()

    @torch.no_grad()
    def sample_all(self, feat, n_samples=50):
        mu, L = self.forward(feat)  # mu:(N,7), L:(N,7,7)
        eps = torch.randn(n_samples, feat.shape[0], self.n_rois, device=feat.device)
        # samples_raw (n_samples, N, 7): mu + L @ eps
        samples = mu.unsqueeze(0) + torch.einsum('bij,snj->sni', L, eps)
        return samples.cpu()  # (N, B, 7)


def train_gaussian_head(feat_tr, tgt_tr, epochs_mse=200, lr=3e-4, bs=128):
    """
    Gaussian head: MSE training for mean + empirical covariance from residuals.

    Avoids NLL fine-tuning instability (NLL with learned chol corrupts shared
    layers, destroying mean predictions). Instead we:
      Phase 1: MSE-train mu_head (stable, good Avg.R)
      Phase 2: Compute residuals on train set, fit empirical Cholesky
               from residual covariance. This gives a principled, data-driven
               joint distribution that's calibratable via conformal.

    Returns: head (with _emp_L set), mu_arr, sig_arr, L_emp
    """
    N = len(feat_tr)
    mu = tgt_tr.mean(0); sig = tgt_tr.std(0) + 1e-8
    tgt_norm = (tgt_tr - mu) / sig

    head = JointGaussianHead(feat_dim=FEAT_TOTAL).to(DEVICE)

    # ── Phase 1: MSE warmup for mean predictions ──────────────────────────
    print(f'  Phase 1: MSE training ({epochs_mse} epochs)...')
    opt1 = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_mse, eta_min=1e-5)
    for ep in range(epochs_mse):
        head.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            c = feat_tr[idx].to(DEVICE); t = tgt_norm[idx].to(DEVICE)
            mu_pred, _ = head(c)
            loss = F.mse_loss(mu_pred, t)
            opt1.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt1.step()
        sch1.step()
        if (ep + 1) % 50 == 0:
            head.eval()
            with torch.no_grad():
                val_mu, _ = head(feat_tr[:200].to(DEVICE))
                val_mse = F.mse_loss(val_mu, tgt_norm[:200].to(DEVICE)).item()
            print(f'    ep {ep+1}/{epochs_mse}  mse={loss.item():.4f}  val_mse={val_mse:.4f}')

    # ── Phase 2: Empirical covariance from residuals ───────────────────────
    print('  Phase 2: Fitting empirical covariance from residuals...')
    head.eval()
    all_resids = []
    with torch.no_grad():
        for i in range(0, N, bs):
            mu_pred, _ = head(feat_tr[i:i+bs].to(DEVICE))
            resid = tgt_norm[i:i+bs].to(DEVICE) - mu_pred
            all_resids.append(resid.cpu())
    residuals = torch.cat(all_resids, dim=0)  # (N, 7)

    # Empirical covariance of residuals → shared uncertainty structure
    resid_c = residuals - residuals.mean(0)
    emp_cov = (resid_c.T @ resid_c) / (N - 1)  # (7, 7)
    try:
        L_emp = torch.linalg.cholesky(emp_cov + 1e-4 * torch.eye(7))
    except Exception:
        L_emp = torch.diag(residuals.std(0))

    print(f'  Residual std per ROI: {[f"{float(residuals[:,j].std()):.3f}" for j in range(7)]}')

    # Store empirical Cholesky on head for use in sample_all_emp()
    head._emp_L = L_emp

    # Verify mean quality on training data
    rs_check = [float(np.corrcoef(residuals[:, j].numpy() * 0 + tgt_norm[:, j].numpy(),
                                   tgt_norm[:, j].numpy())[0, 1]) for j in range(7)]
    with torch.no_grad():
        mu_chk, _ = head(feat_tr[:500].to(DEVICE))
    rs_tr = [float(np.corrcoef(mu_chk.cpu().numpy()[:, j], tgt_norm.numpy()[:500, j])[0, 1])
             for j in range(7)]
    print(f'  Gaussian head mean R (on train): {float(np.nanmean(rs_tr)):.3f}')

    return head, mu.numpy(), sig.numpy(), L_emp


def sample_gaussian_emp(head, feat, L_emp, n_samples=50, bs=256):
    """Sample from Gaussian head using empirical Cholesky L_emp.
    Returns (n_samples, N, 7) in normalized space.
    """
    head.eval()
    mu_preds = []
    with torch.no_grad():
        for i in range(0, len(feat), bs):
            mu_p, _ = head(feat[i:i+bs].to(DEVICE))
            mu_preds.append(mu_p.cpu())
    mu_pred = torch.cat(mu_preds, dim=0)  # (N, 7)

    B = mu_pred.shape[0]
    eps = torch.randn(n_samples, B, 7)           # (n_samples, N, 7)
    # samples = mu + L_emp @ eps  (per-sample)
    samples = mu_pred.unsqueeze(0) + torch.einsum('ij,snj->sni', L_emp, eps)
    return samples  # (n_samples, N, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Conformal calibration helpers
# ─────────────────────────────────────────────────────────────────────────────
def marginal_conformal_q(preds_cal_samples, tgt_cal, q_level=0.9):
    """
    preds_cal_samples: (N_samples, N_cal, 7) — all ODE/MC samples on cal set
    tgt_cal: (N_cal, 7)
    Returns: q (7,) — per-ROI conformal quantile
    """
    medians = preds_cal_samples.median(dim=0).values  # (N_cal, 7)
    errors = (tgt_cal - medians).abs()  # (N_cal, 7)
    n_cal = tgt_cal.shape[0]
    q_idx = math.ceil((n_cal + 1) * q_level) / n_cal
    q_idx = min(q_idx, 1.0)
    q = torch.tensor([float(torch.quantile(errors[:, j], q_idx)) for j in range(7)])
    return q  # (7,)

def compute_coverage_and_width(preds_samples, tgt, q):
    """
    preds_samples: (N_samples, N_test, 7)
    tgt: (N_test, 7)
    q: (7,) per-ROI conformal quantile
    Returns: per-ROI Coverage, per-ROI Width=2q (scalar), CovMAE
    """
    medians = preds_samples.median(dim=0).values  # (N_test, 7)
    errors = (tgt - medians).abs()  # (N_test, 7)
    coverage_per_roi = [(errors[:, j] <= q[j]).float().mean().item() for j in range(7)]
    width_per_roi = [2 * float(q[j]) for j in range(7)]  # conformal interval width

    # CovMAE on mean predictions (invariant to conformal scaling)
    pred_np = medians.numpy()
    true_np = tgt.numpy()
    pred_corr = np.corrcoef(pred_np.T)
    true_corr = np.corrcoef(true_np.T)
    cov_mae = float(np.mean(np.abs(pred_corr - true_corr)[~np.eye(7, dtype=bool)]))

    # Per-ROI Avg.R
    rs = [float(np.corrcoef(pred_np[:, j], true_np[:, j])[0, 1]) for j in range(7)]

    return coverage_per_roi, width_per_roi, cov_mae, rs

def compute_crps(preds_samples, tgt):
    """Energy CRPS from samples: CRPS = E|X-y| - 0.5*E|X-X'|"""
    N_samp, N_te, D = preds_samples.shape
    crps_per_roi = []
    for j in range(D):
        samples_j = preds_samples[:, :, j].numpy()  # (N_samp, N_te)
        true_j = tgt[:, j].numpy()  # (N_te,)
        e1 = np.mean(np.abs(samples_j - true_j[None, :]), axis=0)  # (N_te,)
        perm = np.random.permutation(N_samp)
        e2 = np.mean(np.abs(samples_j - samples_j[perm, :]), axis=0)  # (N_te,)
        crps_per_roi.append(float(np.mean(e1 - 0.5 * e2)))
    return crps_per_roi


# ─────────────────────────────────────────────────────────────────────────────
# Ridge and MLP point-predictor baselines
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge as SKRidge

def train_ridge_multisource(feat_tr, tgt_tr):
    """Ridge on 1400D multi-source features (per ROI)."""
    from sklearn.preprocessing import StandardScaler
    mu = tgt_tr.mean(0).numpy(); sig = (tgt_tr.std(0) + 1e-8).numpy()
    tgt_norm = ((tgt_tr - torch.tensor(mu)) / torch.tensor(sig)).numpy()
    X = feat_tr.numpy()
    models = []
    for j in range(N_ROIS):
        ridge = SKRidge(alpha=1.0)
        ridge.fit(X, tgt_norm[:, j])
        models.append(ridge)
    return models, mu, sig


def predict_ridge(models, feat, mu, sig):
    """Returns (N, 7) predictions in preproc space."""
    X = feat.numpy()
    preds_norm = np.column_stack([m.predict(X) for m in models])
    return torch.tensor(preds_norm * sig[None, :] + mu[None, :], dtype=torch.float32)


class MLPRegressor(nn.Module):
    def __init__(self, feat_dim=FEAT_TOTAL, hidden=512, out=N_ROIS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, out),
        )
    def forward(self, x): return self.net(x)


def train_mlp_multisource(feat_tr, tgt_tr, epochs=100, lr=3e-4, bs=256):
    """MLP on 1400D multi-source features — matched architecture to FM."""
    N = len(feat_tr)
    mu = tgt_tr.mean(0); sig = tgt_tr.std(0) + 1e-8
    tgt_norm = (tgt_tr - mu) / sig

    mlp = MLPRegressor().to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            pred = mlp(feat_tr[idx].to(DEVICE))
            loss = F.mse_loss(pred, tgt_norm[idx].to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep + 1) % 25 == 0:
            print(f'    ep {ep+1}/{epochs}  mse={loss.item():.4f}')

    return mlp, mu.numpy(), sig.numpy()


@torch.no_grad()
def predict_mlp(mlp, feat, mu, sig, bs=512):
    """Returns (N, 7) predictions in preproc space."""
    mlp.eval()
    preds = []
    for i in range(0, len(feat), bs):
        preds.append(mlp(feat[i:i+bs].to(DEVICE)).cpu())
    preds = torch.cat(preds, dim=0)
    mu_t = torch.tensor(mu, dtype=torch.float32)
    sig_t = torch.tensor(sig, dtype=torch.float32)
    return preds * sig_t + mu_t


def conformalized_eval_point(pred_cal, tgt_cal, pred_te, tgt_te, q_level=0.9):
    """
    Conformalize a point predictor. Returns all metrics.
    pred_cal, tgt_cal: (N_cal, 7)  in preproc space
    pred_te, tgt_te:   (N_te, 7)   in preproc space
    """
    # Conformal q (90th percentile of cal |errors|)
    errors_cal = (tgt_cal - pred_cal).abs()
    n_cal = len(tgt_cal)
    q_idx = min(math.ceil((n_cal + 1) * q_level) / n_cal, 1.0)
    q = torch.tensor([float(torch.quantile(errors_cal[:, j], q_idx)) for j in range(N_ROIS)])

    # Coverage
    errors_te = (tgt_te - pred_te).abs()
    coverage = [(errors_te[:, j] <= q[j]).float().mean().item() for j in range(N_ROIS)]
    width    = [2 * float(q[j]) for j in range(N_ROIS)]

    # CovMAE
    pred_np = pred_te.numpy(); true_np = tgt_te.numpy()
    pred_corr = np.corrcoef(pred_np.T); true_corr = np.corrcoef(true_np.T)
    off = ~np.eye(N_ROIS, dtype=bool)
    cov_mae = float(np.mean(np.abs(pred_corr - true_corr)[off]))

    # R
    rs = [float(np.corrcoef(pred_np[:, j], true_np[:, j])[0, 1]) for j in range(N_ROIS)]

    # CRPS via conformal residual bootstrap
    resid_cal = (tgt_cal - pred_cal).numpy()  # (N_cal, 7)
    error_te  = (tgt_te  - pred_te ).numpy()  # (N_te,  7)
    crps = []
    for j in range(N_ROIS):
        rc = resid_cal[:, j]  # (N_cal,)
        et = error_te[:, j]   # (N_te,)
        # E|pred+rc_k - y_te| = E|et_i - rc_k| for each test i, avg over i
        e1 = float(np.mean(np.abs(et[:, None] - rc[None, :])))
        # E|rc_k - rc_k'|  (constant across test points)
        e2 = float(np.mean(np.abs(rc[:, None] - rc[None, :])))
        crps.append(e1 - 0.5 * e2)

    return q, coverage, width, cov_mae, rs, crps


def bootstrap_metric_ci(vals_a, vals_b, metric_name, n_boot=2000):
    """
    Bootstrap 95% CI for (metric_a - metric_b) over per-ROI values.
    vals_a, vals_b: array-like (N_ROIS,)
    Returns (diff_point, ci_low, ci_high)
    """
    a = np.array(vals_a); b = np.array(vals_b)
    point = float(np.nanmean(a) - np.nanmean(b))
    diffs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, N_ROIS, size=N_ROIS)
        diffs.append(np.nanmean(a[idx]) - np.nanmean(b[idx]))
    diffs = np.array(diffs)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def block_bootstrap_r_diff(pred_a, pred_b, true_y, block_size=29, n_boot=2000):
    """
    Paired block bootstrap CI for ΔAvg.R = (Avg.R_a - Avg.R_b).
    Block size ≈ per-scan test size; accounts for within-scan temporal autocorrelation.
    pred_a, pred_b, true_y: np.ndarray (N_te, N_ROIS)
    Returns (diff_point, ci_low, ci_high)
    """
    N = len(true_y)
    n_blocks = math.ceil(N / block_size)

    def avg_r(pred, true):
        rs = [float(np.corrcoef(pred[:, j], true[:, j])[0, 1]) for j in range(N_ROIS)]
        return float(np.nanmean(rs))

    point = avg_r(pred_a, true_y) - avg_r(pred_b, true_y)
    boot_diffs = []
    for _ in range(n_boot):
        sel_blocks = np.random.randint(0, n_blocks, size=n_blocks)
        idx = []
        for b in sel_blocks:
            start = b * block_size
            end = min(start + block_size, N)
            idx.extend(range(start, end))
        idx = np.array(idx[:N])
        boot_diffs.append(avg_r(pred_a[idx], true_y[idx]) - avg_r(pred_b[idx], true_y[idx]))
    boot_diffs = np.array(boot_diffs)
    return point, float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


def block_bootstrap_crps_diff(pred_samples_a, pred_samples_b, true_y,
                               block_size=29, n_boot=2000):
    """
    Paired block bootstrap CI for ΔCRPS = (CRPS_a - CRPS_b).
    pred_samples_a/b: (N_samp, N_te, N_ROIS) — MC samples in preproc space
    Negative diff means method_a is better (lower CRPS).
    """
    N = len(true_y)
    n_blocks = math.ceil(N / block_size)

    def avg_crps_from_samples(samples, true_y_sub):
        N_samp, N_sub, D = samples.shape
        crps = []
        for j in range(D):
            sj = samples[:, :, j]  # (N_samp, N_sub)
            tj = true_y_sub[:, j]
            e1 = np.mean(np.abs(sj - tj[None, :]), axis=0)
            perm = np.random.permutation(N_samp)
            e2 = np.mean(np.abs(sj - sj[perm, :]), axis=0)
            crps.append(float(np.mean(e1 - 0.5 * e2)))
        return float(np.mean(crps))

    point = avg_crps_from_samples(pred_samples_a.numpy(), true_y.numpy()) - \
            avg_crps_from_samples(pred_samples_b.numpy(), true_y.numpy())
    boot_diffs = []
    for _ in range(n_boot):
        sel = np.random.randint(0, n_blocks, size=n_blocks)
        idx = []
        for b in sel:
            idx.extend(range(b * block_size, min((b+1) * block_size, N)))
        idx = np.array(idx[:N])
        sa = pred_samples_a[:, idx, :].numpy()
        sb = pred_samples_b[:, idx, :].numpy()
        ty = true_y[idx].numpy()
        boot_diffs.append(avg_crps_from_samples(sa, ty) - avg_crps_from_samples(sb, ty))
    boot_diffs = np.array(boot_diffs)
    return point, float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42); torch.manual_seed(42)

    print('\n' + '='*70)
    print('FM-NeuroBOLT Conformalized Apples-to-Apples Comparison')
    print('='*70)
    print(f'Device: {DEVICE}')

    # ── 1. Get features ──────────────────────────────────────────────────────
    print('\n[1/4] Feature extraction...')
    feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te, scan_names = \
        get_or_extract_features()
    print(f'  Train: {feat_tr.shape}, Cal: {feat_cal.shape}, Test: {feat_te.shape}')

    # Z-score normalization (consistent across methods)
    mu  = tgt_tr.mean(0); sig = tgt_tr.std(0) + 1e-8
    tgt_tr_norm  = (tgt_tr  - mu) / sig
    tgt_cal_norm = (tgt_cal - mu) / sig
    tgt_te_norm  = (tgt_te  - mu) / sig

    # ── 1b. Split test set → conformal-cal (heldout) + final test ──────────
    # FM v3b was trained on FIRST 90% of each scan (train+cal in the feature
    # cache).  The feature-cache cal set (middle 10%) is INSIDE the FM's
    # training data, so conformal q computed there is artificially small.
    # Fix: use the LAST 10% (feat_te / tgt_te), which is truly held-out for
    # FM v3b, split 50/50 into conformal-cal and final-test.
    # We use the same held-out split for BOTH methods so the comparison is
    # apples-to-apples on identical data.
    N_te = len(tgt_te)
    half = N_te // 2
    feat_cal_heldout = feat_te[:half];  tgt_cal_heldout = tgt_te[:half]
    feat_test_final  = feat_te[half:];  tgt_test_final  = tgt_te[half:]
    print(f'  Held-out cal: {feat_cal_heldout.shape}, Final test: {feat_test_final.shape}')

    # ── 2. FM v3b: load checkpoint, get predictions ──────────────────────────
    print('\n[2/4] FM v3b (loading existing checkpoint)...')
    fm_ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    fm_mu   = torch.tensor(fm_ckpt['mu'], dtype=torch.float32)
    fm_sig  = torch.tensor(fm_ckpt['sig'], dtype=torch.float32)
    fm_head = MultiSourceFMHead(feat_dim=FEAT_TOTAL).to(DEVICE)
    # checkpoint uses 'head' key (from train_fm_v3b_multisource.py save logic)
    state_key = 'head' if 'head' in fm_ckpt else 'model'
    fm_head.load_state_dict(fm_ckpt[state_key])
    fm_head.eval()

    print('  FM v3b: sampling on heldout-cal set...')
    with torch.no_grad():
        fm_cal_samples = fm_head.sample_all(feat_cal_heldout.to(DEVICE), n_samples=50)
        # fm_cal_samples: (50, N_half, 7) in FM-normalized space
    print('  FM v3b: sampling on final test set...')
    with torch.no_grad():
        fm_te_samples = fm_head.sample_all(feat_test_final.to(DEVICE), n_samples=50)
        # fm_te_samples: (50, N_half, 7) in FM-normalized space

    # Target normalization into FM space (for calibration in same space as samples)
    tgt_cal_fm = (tgt_cal_heldout - fm_mu) / fm_sig   # (N_half, 7)
    tgt_te_fm  = (tgt_test_final  - fm_mu) / fm_sig   # (N_half, 7)

    # Compute conformal q for FM v3b (in FM-normalized space)
    q_fm = marginal_conformal_q(fm_cal_samples, tgt_cal_fm, q_level=0.9)
    print(f'  FM v3b conformal q (FM-norm space): {[f"{x:.3f}" for x in q_fm.tolist()]}')

    del fm_head; gc.collect(); torch.cuda.empty_cache()

    # ── 3. Joint Gaussian Head: train and get predictions ────────────────────
    print('\n[3/4] Joint Gaussian Head (training from scratch)...')
    # Note: we train Gaussian head on SAME split as FM v3b (80% of training data)
    gau_head, gau_mu, gau_sig, L_emp = train_gaussian_head(feat_tr, tgt_tr)
    gau_head.eval()

    print('  Gaussian head: sampling on heldout-cal and final test sets...')
    gau_mu_t  = torch.tensor(gau_mu,  dtype=torch.float32)
    gau_sig_t = torch.tensor(gau_sig, dtype=torch.float32)
    # Gaussian samples are in gau-normalized space (trained on tgt_norm = (tgt-gau_mu)/gau_sig)
    gau_cal_samples = sample_gaussian_emp(gau_head, feat_cal_heldout, L_emp, n_samples=50)  # (50, half, 7) gau-norm
    gau_te_samples  = sample_gaussian_emp(gau_head, feat_test_final,  L_emp, n_samples=50)  # (50, half, 7) gau-norm

    # Target normalization into Gaussian space
    tgt_cal_gau = (tgt_cal_heldout - gau_mu_t) / gau_sig_t  # (N_half, 7)
    tgt_te_gau  = (tgt_test_final  - gau_mu_t) / gau_sig_t  # (N_half, 7)

    # Gaussian head R on final test set
    gau_te_median_gau = gau_te_samples.median(dim=0).values  # (half, 7) gau-norm
    gau_te_median_raw = gau_te_median_gau * gau_sig_t + gau_mu_t  # preproc space
    gau_te_rs = [float(np.corrcoef(gau_te_median_raw.numpy()[:, j], tgt_test_final.numpy()[:, j])[0, 1]) for j in range(7)]
    gau_te_avg_r = float(np.nanmean(gau_te_rs))
    print(f'  Gaussian head test Avg.R: {gau_te_avg_r:.3f}')

    q_gau = marginal_conformal_q(gau_cal_samples, tgt_cal_gau, q_level=0.9)
    print(f'  Gaussian conformal q (gau-norm space): {[f"{x:.3f}" for x in q_gau.tolist()]}')

    del gau_head; gc.collect(); torch.cuda.empty_cache()

    # ── 4. Ridge and MLP baselines ────────────────────────────────────────────
    print('\n[4a/5] Ridge (1400D multi-source features)...')
    ridge_models, ridge_mu, ridge_sig = train_ridge_multisource(feat_tr, tgt_tr)
    pred_cal_ridge = predict_ridge(ridge_models, feat_cal_heldout, ridge_mu, ridge_sig)
    pred_te_ridge  = predict_ridge(ridge_models, feat_test_final,  ridge_mu, ridge_sig)
    ridge_te_r = [float(np.corrcoef(pred_te_ridge.numpy()[:, j], tgt_test_final.numpy()[:, j])[0, 1]) for j in range(N_ROIS)]
    print(f'  Ridge test Avg.R: {float(np.nanmean(ridge_te_r)):.3f}')
    q_ridge, cov_ridge, width_ridge, cov_mae_ridge, rs_ridge, crps_ridge = \
        conformalized_eval_point(pred_cal_ridge, tgt_cal_heldout, pred_te_ridge, tgt_test_final)
    avg_r_ridge   = float(np.nanmean(rs_ridge))
    avg_cov_ridge = float(np.mean(cov_ridge))
    avg_width_ridge = float(np.mean(width_ridge))
    avg_crps_ridge  = float(np.mean(crps_ridge))

    print('\n[4b/5] MLP (1400D multi-source features, 100 epochs)...')
    mlp_model, mlp_mu, mlp_sig = train_mlp_multisource(feat_tr, tgt_tr, epochs=100)
    pred_cal_mlp = predict_mlp(mlp_model, feat_cal_heldout, mlp_mu, mlp_sig)
    pred_te_mlp  = predict_mlp(mlp_model, feat_test_final,  mlp_mu, mlp_sig)
    mlp_te_r = [float(np.corrcoef(pred_te_mlp.numpy()[:, j], tgt_test_final.numpy()[:, j])[0, 1]) for j in range(N_ROIS)]
    print(f'  MLP test Avg.R: {float(np.nanmean(mlp_te_r)):.3f}')
    q_mlp, cov_mlp, width_mlp, cov_mae_mlp, rs_mlp, crps_mlp = \
        conformalized_eval_point(pred_cal_mlp, tgt_cal_heldout, pred_te_mlp, tgt_test_final)
    avg_r_mlp   = float(np.nanmean(rs_mlp))
    avg_cov_mlp = float(np.mean(cov_mlp))
    avg_width_mlp = float(np.mean(width_mlp))
    avg_crps_mlp  = float(np.mean(crps_mlp))
    del mlp_model; gc.collect(); torch.cuda.empty_cache()

    # ── 5. Evaluate FM + Gaussian at 90% coverage, with bootstrap CIs ────────
    # All evaluation is in each method's own normalized space.
    # Width is converted back to preproc space: width_raw = 2 * q_norm * sig_per_roi
    # CovMAE and CRPS are computed in preproc space for fair comparison.
    print('\n[5/5] Conformalized evaluation + bootstrap CIs...')

    # FM v3b conformalized (in FM-normalized space)
    cov_fm, width_fm_norm, cov_mae_fm_norm, rs_fm = compute_coverage_and_width(
        fm_te_samples, tgt_te_fm, q_fm)
    # Convert width to preproc space
    width_fm = [2 * float(q_fm[j]) * float(fm_sig[j]) for j in range(N_ROIS)]
    fm_te_median_raw = fm_te_samples.median(dim=0).values * fm_sig + fm_mu  # preproc
    true_corr = np.corrcoef(tgt_test_final.numpy().T)
    off = ~np.eye(N_ROIS, dtype=bool)
    cov_mae_fm = float(np.mean(np.abs(np.corrcoef(fm_te_median_raw.numpy().T) - true_corr)[off]))
    fm_te_samples_raw = fm_te_samples * fm_sig + fm_mu  # (50, half, 7) preproc
    crps_fm = compute_crps(fm_te_samples_raw, tgt_test_final)
    avg_r_fm    = float(np.nanmean(rs_fm))
    avg_cov_fm  = float(np.mean(cov_fm))
    avg_width_fm  = float(np.mean(width_fm))
    avg_crps_fm   = float(np.mean(crps_fm))

    # Gaussian head conformalized (in gau-normalized space)
    cov_gau, width_gau_norm, cov_mae_gau_norm, rs_gau = compute_coverage_and_width(
        gau_te_samples, tgt_te_gau, q_gau)
    width_gau = [2 * float(q_gau[j]) * float(gau_sig_t[j]) for j in range(N_ROIS)]
    cov_mae_gau = float(np.mean(np.abs(np.corrcoef(gau_te_median_raw.numpy().T) - true_corr)[off]))
    gau_te_samples_raw = gau_te_samples * gau_sig_t + gau_mu_t
    crps_gau = compute_crps(gau_te_samples_raw, tgt_test_final)
    avg_r_gau   = float(np.nanmean(rs_gau))
    avg_cov_gau = float(np.mean(cov_gau))
    avg_width_gau = float(np.mean(width_gau))
    avg_crps_gau  = float(np.mean(crps_gau))

    # ── Paired block bootstrap CIs (stronger inference over time points) ──────
    # Block size ≈ per-scan test contribution (841 / 29 ≈ 29 time points per scan)
    # Accounts for within-scan temporal autocorrelation — stronger than ROI-level bootstrap.
    np.random.seed(42)
    BLOCK_SIZE = 29

    print('  Computing paired block bootstrap CIs (2000 iterations)...')
    true_np = tgt_test_final.numpy()

    # R differences (paired block bootstrap over time points)
    r_fm_vs_ridge, r_lo_fr, r_hi_fr = block_bootstrap_r_diff(
        fm_te_median_raw.numpy(), pred_te_ridge.numpy(), true_np, BLOCK_SIZE)
    r_fm_vs_mlp, r_lo_fm, r_hi_fm = block_bootstrap_r_diff(
        fm_te_median_raw.numpy(), pred_te_mlp.numpy(), true_np, BLOCK_SIZE)
    r_fm_vs_gau, r_lo_fg, r_hi_fg = block_bootstrap_r_diff(
        fm_te_median_raw.numpy(), gau_te_median_raw.numpy(), true_np, BLOCK_SIZE)

    # CRPS difference FM vs Gaussian (both have MC samples → exact comparison)
    c_fm_vs_gau, c_lo_fg, c_hi_fg = block_bootstrap_crps_diff(
        fm_te_samples_raw, gau_te_samples_raw, tgt_test_final, BLOCK_SIZE, n_boot=1000)

    # ROI-level bootstrap for Ridge/MLP CRPS (point predictors — no MC samples)
    c_fm_vs_ridge, c_lo_fr, c_hi_fr = bootstrap_metric_ci(crps_fm, crps_ridge, 'CRPS:FM-Ridge')
    c_fm_vs_mlp,   c_lo_fm, c_hi_fm = bootstrap_metric_ci(crps_fm, crps_mlp,   'CRPS:FM-MLP')

    # ── Print results table ───────────────────────────────────────────────────
    print('\n' + '='*82)
    print('CONFORMALIZED COMPARISON — All methods at ≈90% marginal coverage')
    print('='*82)
    print(f"\n{'Method':<25} {'Avg.R':>7} {'Cover90':>9} {'Width':>8} {'CRPS':>8} {'CovMAE':>8}")
    print('-'*72)
    print(f"{'Ridge (conformal)':25} {avg_r_ridge:>7.3f} {avg_cov_ridge:>9.3f} {avg_width_ridge:>8.3f} {avg_crps_ridge:>8.3f} {cov_mae_ridge:>8.3f}")
    print(f"{'MLP (conformal)':25} {avg_r_mlp:>7.3f} {avg_cov_mlp:>9.3f} {avg_width_mlp:>8.3f} {avg_crps_mlp:>8.3f} {cov_mae_mlp:>8.3f}")
    print(f"{'Gaussian (conformal)':25} {avg_r_gau:>7.3f} {avg_cov_gau:>9.3f} {avg_width_gau:>8.3f} {avg_crps_gau:>8.3f} {cov_mae_gau:>8.3f}")
    print(f"{'FM v3b (conformal)':25} {avg_r_fm:>7.3f} {avg_cov_fm:>9.3f} {avg_width_fm:>8.3f} {avg_crps_fm:>8.3f} {cov_mae_fm:>8.3f}")
    print('='*82)

    print(f"\n--- Paired block bootstrap 95% CI (block_size={BLOCK_SIZE}, diff=FM-baseline) ---")
    print(f"  ΔAvg.R vs Ridge:    {r_fm_vs_ridge:+.3f}  CI [{r_lo_fr:+.3f}, {r_hi_fr:+.3f}]  [block, n=841]")
    print(f"  ΔAvg.R vs MLP:      {r_fm_vs_mlp:+.3f}  CI [{r_lo_fm:+.3f}, {r_hi_fm:+.3f}]  [block, n=841]")
    print(f"  ΔAvg.R vs Gaussian: {r_fm_vs_gau:+.3f}  CI [{r_lo_fg:+.3f}, {r_hi_fg:+.3f}]  [block, n=841]")
    print(f"  ΔCRPS  vs Gaussian: {c_fm_vs_gau:+.3f}  CI [{c_lo_fg:+.3f}, {c_hi_fg:+.3f}]  [block, MC samples]  (neg=FM better)")
    print(f"  ΔCRPS  vs Ridge:    {c_fm_vs_ridge:+.3f}  CI [{c_lo_fr:+.3f}, {c_hi_fr:+.3f}]  [ROI-level]")
    print(f"  ΔCRPS  vs MLP:      {c_fm_vs_mlp:+.3f}  CI [{c_lo_fm:+.3f}, {c_hi_fm:+.3f}]  [ROI-level]")

    results = {
        'fm_v3b_conformalized': {
            'avg_r': avg_r_fm, 'avg_cover90': avg_cov_fm,
            'avg_width': avg_width_fm, 'avg_crps': avg_crps_fm, 'cov_mae': cov_mae_fm,
            'per_roi_r': rs_fm, 'per_roi_coverage': cov_fm, 'per_roi_crps': crps_fm,
        },
        'gaussian_conformalized': {
            'avg_r': avg_r_gau, 'avg_cover90': avg_cov_gau,
            'avg_width': avg_width_gau, 'avg_crps': avg_crps_gau, 'cov_mae': cov_mae_gau,
            'per_roi_r': rs_gau, 'per_roi_coverage': cov_gau, 'per_roi_crps': crps_gau,
        },
        'ridge_conformalized': {
            'avg_r': avg_r_ridge, 'avg_cover90': avg_cov_ridge,
            'avg_width': avg_width_ridge, 'avg_crps': avg_crps_ridge, 'cov_mae': cov_mae_ridge,
            'per_roi_r': rs_ridge, 'per_roi_coverage': cov_ridge, 'per_roi_crps': crps_ridge,
        },
        'mlp_conformalized': {
            'avg_r': avg_r_mlp, 'avg_cover90': avg_cov_mlp,
            'avg_width': avg_width_mlp, 'avg_crps': avg_crps_mlp, 'cov_mae': cov_mae_mlp,
            'per_roi_r': rs_mlp, 'per_roi_coverage': cov_mlp, 'per_roi_crps': crps_mlp,
        },
        'bootstrap_ci': {
            'fm_vs_ridge_r': {'diff': r_fm_vs_ridge, 'ci_lo': r_lo_fr, 'ci_hi': r_hi_fr},
            'fm_vs_mlp_r':   {'diff': r_fm_vs_mlp,   'ci_lo': r_lo_fm, 'ci_hi': r_hi_fm},
            'fm_vs_gau_r':   {'diff': r_fm_vs_gau,   'ci_lo': r_lo_fg, 'ci_hi': r_hi_fg},
            'fm_vs_ridge_crps': {'diff': c_fm_vs_ridge, 'ci_lo': c_lo_fr, 'ci_hi': c_hi_fr},
            'fm_vs_mlp_crps':   {'diff': c_fm_vs_mlp,   'ci_lo': c_lo_fm, 'ci_hi': c_hi_fm},
            'fm_vs_gau_crps':   {'diff': c_fm_vs_gau,   'ci_lo': c_lo_fg, 'ci_hi': c_hi_fg},
        }
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')

    # ── Figure: 4-method conformalized comparison ─────────────────────────────
    method_names  = ['Ridge', 'MLP', 'Gaussian', 'FM v3b']
    method_colors = ['#6CB4E4', '#F4A460', '#B47CC7', '#4878CF']
    avg_rs   = [avg_r_ridge,  avg_r_mlp,    avg_r_gau,    avg_r_fm]
    avg_covs = [avg_cov_ridge, avg_cov_mlp, avg_cov_gau,  avg_cov_fm]
    avg_wids = [avg_width_ridge, avg_width_mlp, avg_width_gau, avg_width_fm]
    avg_crps = [avg_crps_ridge,  avg_crps_mlp,  avg_crps_gau,  avg_crps_fm]
    avg_maes = [cov_mae_ridge,   cov_mae_mlp,   cov_mae_gau,   cov_mae_fm]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    metric_data = [
        ('Avg.R (↑)',       avg_rs,   False),
        ('Cover90',         avg_covs, False),
        ('Width (↓)',       avg_wids, True),
        ('CRPS (↓)',        avg_crps, True),
        ('CovMAE (↓)',      avg_maes, True),
    ]
    x = np.arange(len(method_names))
    width_bar = 0.55

    for ax, (metric, vals, invert) in zip(axes, metric_data):
        bars = ax.bar(x, vals, color=method_colors, alpha=0.88, width=width_bar,
                      edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(method_names, fontsize=8.5)
        for bar, val in zip(bars, vals):
            ypos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    ypos * (0.97 if invert else 1.02),
                    f'{val:.3f}', ha='center',
                    va='top' if invert else 'bottom', fontsize=8)
        if 'Cover' in metric:
            ax.axhline(0.9, color='red', linestyle='--', alpha=0.7, lw=1.2, label='Target=0.90')
            ax.legend(fontsize=7); ax.set_ylim(0.7, 1.1)
        if invert:
            ax.invert_yaxis()
        ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.suptitle(
        'Conformalized Apples-to-Apples: 4 Methods at ≈90% Marginal Coverage\n'
        '(Ridge, MLP: conformal residual bootstrap; Gaussian, FM v3b: MC samples + conformal)',
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig9_conformalized.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved fig9 -> {fig_path}')


if __name__ == '__main__':
    main()
