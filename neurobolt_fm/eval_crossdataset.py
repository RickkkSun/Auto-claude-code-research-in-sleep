"""
eval_crossdataset.py — FM-NeuroBOLT v3b Cross-Dataset Evaluation

Evaluates the pretrained joint-7D multi-source flow-matching head on external
EEG-fMRI datasets (ds002336, noddi, ds003768) in two modes:

  zeroshot  — apply pretrained FM head directly, no fine-tuning
  finetune  — train FM head from scratch on train subjects (leave-one-out or
              train-frac split), test on held-out subjects

Usage:
  python eval_crossdataset.py --datasets ds002336,noddi,ds003768 --mode zeroshot
  python eval_crossdataset.py --datasets ds002336 --mode finetune --train-frac 0.8

Output: fm_crossdataset_results.json
"""

import sys, os, gc, json, math, warnings, argparse, glob
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
# Global constants (identical to train_fm_v3b_multisource.py)
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
ROI_COLS    = [r[0] for r in ROIS]
ROI_DISPLAY = [r[2] for r in ROIS]
N_ROIS      = 7
FEAT_PER_BB = 200
FEAT_TOTAL  = N_ROIS * FEAT_PER_BB  # 1400

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TMIN  = -16
CROP  = 3200
EVENT = 'R149'

# TR per external dataset (seconds)
DATASET_TR = {
    'ds002336': 2.0,
    'noddi':    2.16,
    'ds003768': 2.1,
    'natview':                2.1,
    'natview-inscapes':       2.1,
    'natview-dme_run-01':     2.1,
    'natview-monkey1_run-01': 2.1,
    'ds002725': 2.0,
    'ds005795': 2.0,
}

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR  = os.path.join(BASE_DIR, 'code', 'checkpoints')
FM_CKPT   = os.path.join(BASE_DIR, 'checkpoints_fm_v3b', 'fm_v3b_joint7d_multisrc.pth')
EXT_DIR   = os.path.join(BASE_DIR, 'external_processed')
OUT_JSON  = os.path.join(BASE_DIR, 'fm_crossdataset_results.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MIN_EPOCHS = 10  # skip subjects with fewer epochs than this


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def to_float(x):
    return float(np.asarray(x).flat[0])


def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# NeuroBOLT backbone
# ─────────────────────────────────────────────────────────────────────────────
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
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def extract_features(backbone, eeg_tensor, bs=64):
    """Extract 200-D features per backbone from EEG tensor (N, 26, 3200)."""
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i + bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b, 'B N A T -> B N (A T)'), input_chans=None)
        out.append(backbone.head_act(xm + xt).cpu())
    return torch.cat(out)  # (N, 200)


# ─────────────────────────────────────────────────────────────────────────────
# FM Head architecture (identical to train_fm_v3b_multisource.py)
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


class MultiSourceFMHead(nn.Module):
    """Joint 7D FM head for 1400-dim multi-source features."""

    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512,
                 n_rois=7, time_dim=32):
        super().__init__()
        self.n_rois   = n_rois
        self.time_emb = SinusoidalEmbedding(time_dim)

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
        )
        self.roi_interact = nn.Linear(n_rois, n_rois)

        in_dim = proj_dim + n_rois + time_dim
        self.velocity_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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
        f_proj = self.feat_proj(features)
        t_emb  = self.time_emb(t)
        x_t_i  = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f_proj, x_t_i, t_emb], dim=-1))

    @torch.no_grad()
    def sample(self, features, n_samples=20, num_steps=100):
        """Monte-Carlo mean: average N independent 7D ODE trajectories."""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x  = torch.randn(B, self.n_rois, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x)
        return torch.stack(preds).mean(0)  # (B, 7)

    @torch.no_grad()
    def sample_ensemble(self, features, n_samples=50, num_steps=100):
        """Return all n_samples trajectories for CRPS computation."""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x  = torch.randn(B, self.n_rois, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x.cpu())
        return torch.stack(preds)  # (n_samples, B, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def discover_subjects(dataset):
    """Return sorted list of subject stem names found in EEG directory."""
    eeg_dir = os.path.join(EXT_DIR, dataset, 'EEG')
    fmri_dir = os.path.join(EXT_DIR, dataset, 'fMRI_difumo64')
    eeg_files = sorted(glob.glob(os.path.join(eeg_dir, '*_eeg.set')))
    subjects = []
    for f in eeg_files:
        stem = os.path.basename(f).replace('_eeg.set', '')
        fmri_f = os.path.join(fmri_dir, f'{stem}_difumo64_roi.pkl')
        if os.path.exists(fmri_f):
            subjects.append(stem)
    return subjects


def load_external_scan(stem, dataset, tr=None):
    """
    Load EEG + all 7 ROI fMRI targets for one external scan.

    Parameters
    ----------
    stem    : str  e.g. 'sub-xp101-scan01'
    dataset : str  e.g. 'ds002336'
    tr      : float  repetition time in seconds; read from processing_log if None

    Returns
    -------
    (eeg, fmri) : (N, 26, 3200) tensor, (N, 7) tensor
    or None if any ROI column is missing or < MIN_EPOCHS epochs.
    """
    if tr is None:
        tr = DATASET_TR.get(dataset, 2.1)

    eeg_path  = os.path.join(EXT_DIR, dataset, 'EEG',           f'{stem}_eeg.set')
    fmri_path = os.path.join(EXT_DIR, dataset, 'fMRI_difumo64', f'{stem}_difumo64_roi.pkl')

    # Load raw EEG
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    # Load fMRI targets
    df = pd.read_pickle(fmri_path)
    b_lp, a_lp = butter(5, 0.15 / (0.5 / tr), btype='low')

    all_fmri = []
    eeg_all  = None

    for col in ROI_COLS:
        if col not in df.columns:
            del raw
            return None
        fm = df[[col]].to_numpy().T            # (1, T_fmri)
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            if len(ep['eeg']) < MIN_EPOCHS:
                del raw
                return None
            eeg_all = torch.stack(
                [torch.tensor(x, dtype=torch.float32) for x in ep['eeg']]
            )
        all_fmri.append(
            torch.tensor([to_float(x) for x in ep['fmri']], dtype=torch.float32)
        )

    del raw
    fmri_all = torch.stack(all_fmri, dim=1)  # (N, 7)
    return eeg_all, fmri_all


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (multi-source, all 7 backbones)
# ─────────────────────────────────────────────────────────────────────────────
def extract_multi_source(eeg_scans_list):
    """
    Given a list of EEG tensors [(N_i, 26, 3200), ...] extract 1400-D features.

    Returns concatenated (N_total, 1400) tensor.
    Loads each backbone once, extracts across all scans, then frees it.
    """
    bb_feats = []   # one (N_total, 200) per backbone

    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'    backbone {j+1}/7: {display}...', flush=True)
        backbone = build_backbone(ckpt_path)

        parts = [extract_features(backbone, eeg) for eeg in eeg_scans_list]
        bb_feats.append(torch.cat(parts, dim=0))   # (N_total, 200)

        del backbone; gc.collect(); torch.cuda.empty_cache()

    return torch.cat(bb_feats, dim=1)   # (N_total, 1400)


# ─────────────────────────────────────────────────────────────────────────────
# CRPS (Continuous Ranked Probability Score)
# ─────────────────────────────────────────────────────────────────────────────
def crps_ensemble(ensemble, obs):
    """
    Compute mean CRPS from an ensemble of predictions.

    Parameters
    ----------
    ensemble : np.ndarray  (n_members, N, n_rois)
    obs      : np.ndarray  (N, n_rois)

    Returns
    -------
    crps_per_roi : np.ndarray  (n_rois,)
    mean_crps    : float
    """
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    # where X, X' are independent draws from the ensemble
    M, N, R = ensemble.shape
    crps_vals = np.zeros(R)
    for r in range(R):
        ens_r = ensemble[:, :, r]   # (M, N)
        obs_r = obs[:, r]           # (N,)
        # E|X - y|
        e1 = np.mean(np.abs(ens_r - obs_r[None, :]), axis=0)   # (N,)
        # 0.5 * E|X - X'|
        # Using the order-statistic identity:
        #   E|X-X'| = (2/M^2) * sum_i (2i - M + 1) * X_{(i)}   (0-indexed i)
        #   => 0.5*E|X-X'| = (1/M^2) * sum_i (2i - M + 1) * X_{(i)}
        sorted_ens = np.sort(ens_r, axis=0)                     # (M, N)
        idx = np.arange(M)
        weights = 2 * idx - M + 1                               # (M,)
        e2_half = np.sum(weights[:, None] * sorted_ens, axis=0) / (M ** 2)  # (N,)
        crps_vals[r] = float(np.mean(e1 - e2_half))
    return crps_vals, float(np.mean(crps_vals))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(head, feat, tgt, mu, sig, n_samples_crps=50, compute_crps=True):
    """
    Run inference and compute Pearson R (+CRPS if requested) for all ROIs.

    Parameters
    ----------
    head      : MultiSourceFMHead (eval mode)
    feat      : (N, 1400) tensor
    tgt       : (N, 7) tensor
    mu, sig   : (7,) numpy arrays for de-normalisation
    n_samples_crps : int  ensemble size for CRPS

    Returns
    -------
    dict with keys 'per_roi', 'avg_r', optionally 'per_roi_crps', 'avg_crps'
    """
    head.eval()
    with torch.no_grad():
        pred_norm = head.sample(feat.to(DEVICE), n_samples=20, num_steps=100)
    pred_norm_np = pred_norm.cpu().numpy()
    pred = pred_norm_np * sig[None, :] + mu[None, :]
    true = tgt.numpy()

    per_roi = {}
    for j, disp in enumerate(ROI_DISPLAY):
        if len(pred) >= 2:
            r, _ = pearsonr(pred[:, j], true[:, j])
        else:
            r = float('nan')
        mse = float(np.mean((pred[:, j] - true[:, j]) ** 2))
        per_roi[disp] = {'R': float(r), 'MSE': float(mse)}

    avg_r = float(np.nanmean([per_roi[d]['R'] for d in ROI_DISPLAY]))
    result = {'per_roi': per_roi, 'avg_r': avg_r}

    if compute_crps:
        try:
            with torch.no_grad():
                ens_norm = head.sample_ensemble(
                    feat.to(DEVICE), n_samples=n_samples_crps, num_steps=100
                )
            # ens_norm: (n_samples, N, 7)
            ens = ens_norm.numpy() * sig[None, None, :] + mu[None, None, :]
            crps_per_roi, mean_crps = crps_ensemble(ens, true)
            per_roi_crps = {ROI_DISPLAY[j]: float(crps_per_roi[j]) for j in range(N_ROIS)}
            result['per_roi_crps'] = per_roi_crps
            result['avg_crps'] = float(mean_crps)
        except Exception as e:
            print(f'    [warn] CRPS failed: {e}', flush=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FM head training (few-shot / finetune)
# ─────────────────────────────────────────────────────────────────────────────
def train_fm_head(feat_train, tgt_train, epochs=200, lr=3e-4, bs=128):
    """Train a fresh MultiSourceFMHead from scratch (OT-CFM)."""
    N = len(feat_train)
    mu  = tgt_train.mean(0)
    sig = tgt_train.std(0) + 1e-8
    tgt_norm = (tgt_train - mu) / sig

    head = MultiSourceFMHead(feat_dim=FEAT_TOTAL).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_ep = max(1, math.ceil(N / bs))
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        ep_loss = 0.0; nb = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            c   = feat_train[idx].to(DEVICE)
            x1  = tgt_norm[idx].to(DEVICE)
            t   = torch.rand(len(c), 1, device=DEVICE)
            x0  = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            loss = F.mse_loss(head(c, x_t, t), x1 - x0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sch.step()
        if (ep + 1) % 50 == 0:
            print(f'      ep {ep+1}/{epochs}  loss={ep_loss/nb:.4f}', flush=True)

    return head, mu.numpy(), sig.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_zeroshot(dataset, subjects):
    """
    Zero-shot: load pretrained FM head, evaluate directly on each subject.
    Returns per-subject and aggregated results.
    """
    print(f'\n  [zero-shot] Loading pretrained FM head from {FM_CKPT}', flush=True)
    ckpt = torch.load(FM_CKPT, map_location='cpu', weights_only=False)
    head = MultiSourceFMHead(feat_dim=FEAT_TOTAL).to(DEVICE)
    head.load_state_dict(ckpt['head'])
    head.eval()
    mu  = np.array(ckpt['mu'],  dtype=np.float32)
    sig = np.array(ckpt['sig'], dtype=np.float32)

    tr = DATASET_TR.get(dataset, 2.1)
    per_subject = {}

    # Collect all EEG for feature extraction (one backbone pass per backbone)
    print(f'  Loading {len(subjects)} subjects...', flush=True)
    eeg_list, fmri_list, valid_subjects = [], [], []
    for stem in subjects:
        print(f'    {stem}...', end=' ', flush=True)
        try:
            result = load_external_scan(stem, dataset, tr=tr)
            if result is None:
                print('SKIP (missing ROI or too few epochs)', flush=True)
                continue
            eeg, fmri = result
            print(f'n={len(eeg)}', flush=True)
            eeg_list.append(eeg)
            fmri_list.append(fmri)
            valid_subjects.append(stem)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    if not valid_subjects:
        print('  No valid subjects found, skipping dataset.', flush=True)
        return None

    print(f'  Extracting multi-source features ({len(valid_subjects)} subjects)...', flush=True)
    feat_all = extract_multi_source(eeg_list)   # (N_total, 1400)
    fmri_all = torch.cat(fmri_list, dim=0)      # (N_total, 7)

    # Build per-subject slices
    slices, idx = [], 0
    for eeg in eeg_list:
        n = len(eeg); slices.append(slice(idx, idx + n)); idx += n

    del eeg_list; gc.collect()

    # Per-subject metrics
    all_rs = {d: [] for d in ROI_DISPLAY}
    for stem, sl in zip(valid_subjects, slices):
        f_sub = feat_all[sl]; t_sub = fmri_all[sl]
        if len(f_sub) < MIN_EPOCHS:
            continue
        m = compute_metrics(head, f_sub, t_sub, mu, sig, compute_crps=True, n_samples_crps=30)
        per_subject[stem] = m
        print(f'    {stem}: avg_R={m["avg_r"]:.3f}', flush=True)
        for d in ROI_DISPLAY:
            all_rs[d].append(m['per_roi'][d]['R'])

    if not per_subject:
        return None

    # Aggregate over all subjects (pooled)
    agg = compute_metrics(head, feat_all, fmri_all, mu, sig, compute_crps=True, n_samples_crps=50)
    agg['per_subject'] = per_subject

    print(f'\n  => Zero-shot pooled avg_R = {agg["avg_r"]:.3f}', flush=True)
    if 'avg_crps' in agg:
        print(f'  => Zero-shot pooled avg_CRPS = {agg["avg_crps"]:.4f}', flush=True)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot / finetune evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_finetune(dataset, subjects, train_frac=0.8, epochs=200,
                 leave_one_out=False):
    """
    Fine-tune: train FM head from scratch on train subjects, test on held-out.

    If leave_one_out=True (activated when ≤ 3 subjects exist), uses LOO scheme.
    Otherwise splits subjects into train/test by train_frac.
    """
    tr = DATASET_TR.get(dataset, 2.1)

    print(f'  Loading {len(subjects)} subjects...', flush=True)
    eeg_list, fmri_list, valid_subjects = [], [], []
    for stem in subjects:
        print(f'    {stem}...', end=' ', flush=True)
        try:
            result = load_external_scan(stem, dataset, tr=tr)
            if result is None:
                print('SKIP (missing ROI or too few epochs)', flush=True)
                continue
            eeg, fmri = result
            print(f'n={len(eeg)}', flush=True)
            eeg_list.append(eeg)
            fmri_list.append(fmri)
            valid_subjects.append(stem)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    if len(valid_subjects) < 2:
        print('  Not enough subjects for finetune evaluation, skipping.', flush=True)
        return None

    # Decide split scheme
    n_subs = len(valid_subjects)
    if leave_one_out or n_subs <= 3:
        # LOO: one fold per subject
        print(f'  Using leave-one-out ({n_subs} folds)...', flush=True)
        test_folds = [[i] for i in range(n_subs)]
        train_folds = [[j for j in range(n_subs) if j != i] for i in range(n_subs)]
    else:
        n_train = max(1, int(round(n_subs * train_frac)))
        n_test  = n_subs - n_train
        if n_test < 1:
            n_test = 1; n_train = n_subs - 1
        train_idx = list(range(n_train))
        test_idx  = list(range(n_train, n_subs))
        train_folds = [train_idx]
        test_folds  = [test_idx]
        print(f'  Split: {n_train} train / {n_test} test subjects', flush=True)

    # Extract multi-source features once for all subjects
    print(f'  Extracting multi-source features...', flush=True)
    feat_per_sub  = []
    for i, eeg in enumerate(eeg_list):
        feat_per_sub.append(None)  # placeholder; will fill below

    # Extract per-backbone across all subjects
    bb_feat_per_sub = [[] for _ in range(n_subs)]   # [sub_idx][bb_idx] = (N, 200)
    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'    backbone {j+1}/7: {display}...', flush=True)
        backbone = build_backbone(ckpt_path)
        for i, eeg in enumerate(eeg_list):
            bb_feat_per_sub[i].append(extract_features(backbone, eeg))
        del backbone; gc.collect(); torch.cuda.empty_cache()

    # Concatenate backbones per subject
    feat_per_sub = [torch.cat(bb_feat_per_sub[i], dim=1) for i in range(n_subs)]  # (N_i, 1400)
    del bb_feat_per_sub, eeg_list; gc.collect()

    # Run folds
    fold_results = []
    for fold_idx, (tr_idx, te_idx) in enumerate(zip(train_folds, test_folds)):
        print(f'\n  Fold {fold_idx+1}/{len(train_folds)}: '
              f'train={[valid_subjects[i] for i in tr_idx]}, '
              f'test={[valid_subjects[i] for i in te_idx]}', flush=True)

        feat_tr = torch.cat([feat_per_sub[i] for i in tr_idx], dim=0)
        fmri_tr = torch.cat([fmri_list[i]   for i in tr_idx], dim=0)
        feat_te = torch.cat([feat_per_sub[i] for i in te_idx], dim=0)
        fmri_te = torch.cat([fmri_list[i]   for i in te_idx], dim=0)

        if len(feat_tr) < MIN_EPOCHS or len(feat_te) < MIN_EPOCHS:
            print('    SKIP: too few samples', flush=True)
            continue

        print(f'    Training FM head: {len(feat_tr)} train, {len(feat_te)} test', flush=True)
        head, mu, sig = train_fm_head(feat_tr, fmri_tr, epochs=epochs)

        metrics = compute_metrics(head, feat_te, fmri_te, mu, sig,
                                  compute_crps=True, n_samples_crps=30)
        metrics['train_subjects'] = [valid_subjects[i] for i in tr_idx]
        metrics['test_subjects']  = [valid_subjects[i] for i in te_idx]
        fold_results.append(metrics)
        print(f'    Fold avg_R={metrics["avg_r"]:.3f}', flush=True)
        if 'avg_crps' in metrics:
            print(f'    Fold avg_CRPS={metrics["avg_crps"]:.4f}', flush=True)

        del head; gc.collect(); torch.cuda.empty_cache()

    if not fold_results:
        return None

    # Aggregate across folds
    all_r = [f['avg_r'] for f in fold_results]
    agg_avg_r = float(np.nanmean(all_r))
    per_roi_r_agg = {}
    for d in ROI_DISPLAY:
        rs = [f['per_roi'][d]['R'] for f in fold_results if d in f['per_roi']]
        per_roi_r_agg[d] = {'R_mean': float(np.nanmean(rs)),
                            'R_std':  float(np.nanstd(rs))}

    result = {
        'folds':      fold_results,
        'avg_r':      agg_avg_r,
        'per_roi':    per_roi_r_agg,
        'n_folds':    len(fold_results),
    }
    if any('avg_crps' in f for f in fold_results):
        crps_vals = [f['avg_crps'] for f in fold_results if 'avg_crps' in f]
        result['avg_crps'] = float(np.nanmean(crps_vals))

    print(f'\n  => Finetune avg_R ({len(fold_results)} folds) = {agg_avg_r:.3f}', flush=True)
    if 'avg_crps' in result:
        print(f'  => Finetune avg_CRPS = {result["avg_crps"]:.4f}', flush=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Intra-subject evaluation (same as original NeuroBOLT-FM paper)
# ─────────────────────────────────────────────────────────────────────────────
def run_intra(dataset, subjects, train_frac=0.8, epochs=200):
    """
    Intra-subject evaluation: for each subject, split epochs 80/20,
    train FM head from scratch on 80%, test on 20%.
    This is directly comparable to the original paper's evaluation.
    """
    tr = DATASET_TR.get(dataset, 2.1)
    print(f'  Loading {len(subjects)} subjects...', flush=True)
    eeg_list, fmri_list, valid_subjects = [], [], []
    for stem in subjects:
        print(f'    {stem}...', end=' ', flush=True)
        try:
            result = load_external_scan(stem, dataset, tr=tr)
            if result is None:
                print('SKIP (missing ROI or too few epochs)', flush=True)
                continue
            eeg, fmri = result
            print(f'n={len(eeg)}', flush=True)
            eeg_list.append(eeg)
            fmri_list.append(fmri)
            valid_subjects.append(stem)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    if not valid_subjects:
        print('  No valid subjects found.', flush=True)
        return None

    n_subs = len(valid_subjects)
    print(f'  Extracting features for {n_subs} subjects...', flush=True)

    # Extract multi-source features per backbone (process all subjects per backbone)
    bb_feat_per_sub = [[] for _ in range(n_subs)]
    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'    backbone {j+1}/7: {display}...', flush=True)
        backbone = build_backbone(ckpt_path)
        for i, eeg in enumerate(eeg_list):
            bb_feat_per_sub[i].append(extract_features(backbone, eeg))
        del backbone; gc.collect(); torch.cuda.empty_cache()

    feat_per_sub = [torch.cat(bb_feat_per_sub[i], dim=1) for i in range(n_subs)]
    del bb_feat_per_sub, eeg_list; gc.collect()

    # Per-subject intra-subject split and train/test
    sub_results = []
    for i, stem in enumerate(valid_subjects):
        feat  = feat_per_sub[i]   # (N, 1400)
        fmri  = fmri_list[i]      # (N, 7)
        N     = len(feat)
        n_tr  = int(N * train_frac)
        n_val = int(N * 0.1)
        n_te  = N - n_tr - n_val
        if n_te < MIN_EPOCHS or n_tr < MIN_EPOCHS:
            print(f'  {stem}: SKIP (n_tr={n_tr} n_te={n_te})', flush=True)
            continue

        # Split: first 80% train, middle 10% discard (buffer), last 10% test
        feat_tr, fmri_tr = feat[:n_tr],  fmri[:n_tr]
        feat_te, fmri_te = feat[-n_te:], fmri[-n_te:]

        print(f'\n  {stem}: train={n_tr}, test={n_te}', flush=True)
        head, mu, sig = train_fm_head(feat_tr, fmri_tr, epochs=epochs)
        metrics = compute_metrics(head, feat_te, fmri_te, mu, sig, compute_crps=True)
        metrics['subject'] = stem
        sub_results.append(metrics)
        print(f'  {stem}: avg_R={metrics["avg_r"]:.3f}', flush=True)
        if 'avg_crps' in metrics:
            print(f'  {stem}: avg_CRPS={metrics["avg_crps"]:.4f}', flush=True)
        del head; gc.collect(); torch.cuda.empty_cache()

    if not sub_results:
        return None

    agg_avg_r = float(np.nanmean([r['avg_r'] for r in sub_results]))
    per_roi_agg = {}
    for d in ROI_DISPLAY:
        rs = [r['per_roi'][d]['R'] for r in sub_results if d in r.get('per_roi', {})]
        per_roi_agg[d] = {'R_mean': float(np.nanmean(rs)) if rs else float('nan'),
                          'R_std':  float(np.nanstd(rs))  if rs else float('nan')}
    result = {'subjects': sub_results, 'avg_r': agg_avg_r, 'per_roi': per_roi_agg,
              'n_subjects': len(sub_results)}
    if any('avg_crps' in r for r in sub_results):
        result['avg_crps'] = float(np.nanmean([r['avg_crps'] for r in sub_results
                                                if 'avg_crps' in r]))

    print(f'\n  => Intra-subject avg_R ({len(sub_results)} subjects) = {agg_avg_r:.3f}', flush=True)
    if 'avg_crps' in result:
        print(f'  => Intra-subject avg_CRPS = {result["avg_crps"]:.4f}', flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pooled intra-subject (identical protocol to original NeuroBOLT-FM paper)
# Pool all subjects' 80% train, evaluate on each subject's 20% test
# ─────────────────────────────────────────────────────────────────────────────
def run_pooled_intra(dataset, subjects, train_frac=0.8, epochs=300):
    """
    Exact replica of original paper's protocol:
    - Split each subject: first 80% → train, last 10% → test (buffer of 10% discarded)
    - Pool ALL subjects' train splits into one training set
    - Train ONE FM head on pooled data
    - Evaluate on each subject's test split independently
    - Report per-subject avg_R and overall mean avg_R
    """
    tr = DATASET_TR.get(dataset, 2.1)
    print(f'  Loading {len(subjects)} subjects...', flush=True)
    eeg_list, fmri_list, valid_subjects = [], [], []
    for stem in subjects:
        print(f'    {stem}...', end=' ', flush=True)
        try:
            result = load_external_scan(stem, dataset, tr=tr)
            if result is None:
                print('SKIP', flush=True); continue
            eeg, fmri = result
            print(f'n={len(eeg)}', flush=True)
            eeg_list.append(eeg); fmri_list.append(fmri)
            valid_subjects.append(stem)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    if not valid_subjects:
        print('  No valid subjects.', flush=True); return None

    n_subs = len(valid_subjects)
    print(f'  Extracting features ({n_subs} subjects)...', flush=True)

    bb_feat_per_sub = [[] for _ in range(n_subs)]
    for j, (_, ckpt_fname, display) in enumerate(ROIS):
        print(f'    backbone {j+1}/7: {display}...', flush=True)
        backbone = build_backbone(os.path.join(CKPT_DIR, ckpt_fname))
        for i, eeg in enumerate(eeg_list):
            bb_feat_per_sub[i].append(extract_features(backbone, eeg))
        del backbone; gc.collect(); torch.cuda.empty_cache()

    feat_per_sub = [torch.cat(bb_feat_per_sub[i], dim=1) for i in range(n_subs)]
    del bb_feat_per_sub, eeg_list; gc.collect()

    # Split each subject
    train_feat, train_fmri = [], []
    test_splits = []   # list of (feat_te, fmri_te) per subject
    n_tr_total = 0
    for i, stem in enumerate(valid_subjects):
        feat = feat_per_sub[i]; fmri = fmri_list[i]
        N = len(feat)
        n_tr = int(N * train_frac)
        n_buf = int(N * 0.1)
        n_te = N - n_tr - n_buf
        if n_te < MIN_EPOCHS or n_tr < 20:
            print(f'  {stem}: SKIP (too short)', flush=True); continue
        train_feat.append(feat[:n_tr]); train_fmri.append(fmri[:n_tr])
        test_splits.append((feat[-n_te:], fmri[-n_te:], stem))
        n_tr_total += n_tr

    if not test_splits:
        return None

    # Pool and train ONE model
    feat_tr = torch.cat(train_feat); fmri_tr = torch.cat(train_fmri)
    print(f'\n  Pooled training: {n_tr_total} samples from {len(test_splits)} subjects', flush=True)
    head, mu, sig = train_fm_head(feat_tr, fmri_tr, epochs=epochs)

    # Evaluate per subject
    sub_results = []
    for (feat_te, fmri_te, stem) in test_splits:
        metrics = compute_metrics(head, feat_te, fmri_te, mu, sig, compute_crps=True)
        metrics['subject'] = stem
        sub_results.append(metrics)
        print(f'  {stem}: n={len(feat_te)}  avg_R={metrics["avg_r"]:.3f}', flush=True)
    del head; gc.collect(); torch.cuda.empty_cache()

    agg_avg_r = float(np.nanmean([r['avg_r'] for r in sub_results]))
    per_roi_agg = {}
    for d in ROI_DISPLAY:
        rs = [r['per_roi'][d]['R'] for r in sub_results if d in r.get('per_roi', {})]
        per_roi_agg[d] = {'R_mean': float(np.nanmean(rs)) if rs else float('nan')}
    result = {'subjects': sub_results, 'avg_r': agg_avg_r, 'per_roi': per_roi_agg,
              'n_subjects': len(sub_results)}
    if any('avg_crps' in r for r in sub_results):
        result['avg_crps'] = float(np.nanmean([r['avg_crps'] for r in sub_results
                                                if 'avg_crps' in r]))
    print(f'\n  => Pooled intra avg_R ({len(sub_results)} subjects) = {agg_avg_r:.3f}', flush=True)
    if 'avg_crps' in result:
        print(f'  => Pooled intra avg_CRPS = {result["avg_crps"]:.4f}', flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ridge Regression baseline (same pooled protocol as FM)
# ─────────────────────────────────────────────────────────────────────────────
def run_ridge_baseline(dataset, subjects, train_frac=0.8, alpha=1.0):
    """
    Ridge Regression baseline using the same NeuroBOLT features (1400-D).
    Same pooled-intra protocol as run_pooled_intra():
    - Split each subject 80/10/10 (train/buf/test)
    - Pool all subjects' train → fit one Ridge per ROI
    - Evaluate on each subject's test split
    Reports Pearson R per ROI. No CRPS (deterministic).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    tr = DATASET_TR.get(dataset, 2.1)
    print(f'  Loading {len(subjects)} subjects...', flush=True)
    eeg_list, fmri_list, valid_subjects = [], [], []
    for stem in subjects:
        print(f'    {stem}...', end=' ', flush=True)
        try:
            result = load_external_scan(stem, dataset, tr=tr)
            if result is None:
                print('SKIP', flush=True); continue
            eeg, fmri = result
            print(f'n={len(eeg)}', flush=True)
            eeg_list.append(eeg); fmri_list.append(fmri)
            valid_subjects.append(stem)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    if not valid_subjects:
        print('  No valid subjects.', flush=True); return None

    n_subs = len(valid_subjects)
    print(f'  Extracting features ({n_subs} subjects)...', flush=True)

    bb_feat_per_sub = [[] for _ in range(n_subs)]
    for j, (_, ckpt_fname, display) in enumerate(ROIS):
        print(f'    backbone {j+1}/7: {display}...', flush=True)
        backbone = build_backbone(os.path.join(CKPT_DIR, ckpt_fname))
        for i, eeg in enumerate(eeg_list):
            bb_feat_per_sub[i].append(extract_features(backbone, eeg))
        del backbone; gc.collect(); torch.cuda.empty_cache()

    feat_per_sub = [torch.cat(bb_feat_per_sub[i], dim=1) for i in range(n_subs)]
    del bb_feat_per_sub, eeg_list; gc.collect()

    # Split each subject 80/10/10
    train_feat, train_fmri = [], []
    test_splits = []
    for i, stem in enumerate(valid_subjects):
        feat = feat_per_sub[i].numpy(); fmri = fmri_list[i].numpy()
        N = len(feat)
        n_tr  = int(N * train_frac)
        n_buf = int(N * 0.1)
        n_te  = N - n_tr - n_buf
        if n_te < MIN_EPOCHS or n_tr < 20:
            print(f'  {stem}: SKIP (too short)', flush=True); continue
        train_feat.append(feat[:n_tr]); train_fmri.append(fmri[:n_tr])
        test_splits.append((feat[-n_te:], fmri[-n_te:], stem))

    if not test_splits:
        return None

    X_tr = np.concatenate(train_feat, axis=0)
    Y_tr = np.concatenate(train_fmri, axis=0)
    print(f'\n  Ridge training: {len(X_tr)} samples, alpha={alpha}', flush=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    # Fit one Ridge per ROI (multi-output Ridge)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr_s, Y_tr)

    # Evaluate per subject
    sub_results = []
    for (feat_te, fmri_te, stem) in test_splits:
        X_te_s = scaler.transform(feat_te)
        pred   = ridge.predict(X_te_s)   # (N_te, 7)
        per_roi = {}
        for j, disp in enumerate(ROI_DISPLAY):
            if len(pred) >= 2:
                r, _ = pearsonr(pred[:, j], fmri_te[:, j])
            else:
                r = float('nan')
            per_roi[disp] = {'R': float(r)}
        avg_r = float(np.nanmean([per_roi[d]['R'] for d in ROI_DISPLAY]))
        sub_results.append({'subject': stem, 'per_roi': per_roi, 'avg_r': avg_r})
        print(f'  {stem}: n={len(feat_te)}  avg_R={avg_r:.3f}', flush=True)

    agg_avg_r = float(np.nanmean([r['avg_r'] for r in sub_results]))
    per_roi_agg = {}
    for d in ROI_DISPLAY:
        rs = [r['per_roi'][d]['R'] for r in sub_results]
        per_roi_agg[d] = {'R_mean': float(np.nanmean(rs))}

    result = {'subjects': sub_results, 'avg_r': agg_avg_r, 'per_roi': per_roi_agg,
              'n_subjects': len(sub_results), 'alpha': alpha}
    print(f'\n  => Ridge avg_R ({len(sub_results)} subjects) = {agg_avg_r:.3f}', flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='FM-NeuroBOLT v3b cross-dataset evaluation'
    )
    parser.add_argument('--datasets', type=str,
                        default='ds002336,noddi,ds003768,natview,natview-inscapes,natview-dme_run-01,natview-monkey1_run-01,ds002725,ds005795',
                        help='Comma-separated dataset names under external_processed/')
    parser.add_argument('--mode', choices=['zeroshot', 'finetune', 'intra', 'pooled', 'ridge', 'all'],
                        default='pooled',
                        help='Evaluation mode: zeroshot | finetune | intra | pooled | ridge | all')
    parser.add_argument('--train-frac', type=float, default=0.8,
                        help='Fraction of subjects used for training (finetune mode)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs per fold (finetune mode)')
    parser.add_argument('--leave-one-out', action='store_true',
                        help='Force leave-one-out scheme in finetune mode')
    parser.add_argument('--out', type=str, default=OUT_JSON,
                        help='Path to output JSON file')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    modes    = (['zeroshot', 'ridge', 'pooled'] if args.mode == 'all'
                else [args.mode])

    print(f'\n{"="*65}')
    print('FM-NeuroBOLT v3b — Cross-Dataset Evaluation')
    print(f'{"="*65}')
    print(f'Device:   {DEVICE}')
    print(f'Datasets: {datasets}')
    print(f'Modes:    {modes}')
    print(f'{"="*65}\n')

    # Load existing results if present
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    else:
        results = {}

    for dataset in datasets:
        ds_dir = os.path.join(EXT_DIR, dataset)
        if not os.path.isdir(ds_dir):
            print(f'[SKIP] {dataset}: directory not found at {ds_dir}')
            continue

        subjects = discover_subjects(dataset)
        if not subjects:
            print(f'[SKIP] {dataset}: no subjects with paired EEG+fMRI found')
            continue

        tr = DATASET_TR.get(dataset, 2.1)
        print(f'\n{"─"*60}')
        print(f'Dataset: {dataset}  (TR={tr}s,  {len(subjects)} subjects)')
        print(f'Subjects: {subjects}')
        print(f'{"─"*60}')

        if dataset not in results:
            results[dataset] = {}

        for mode in modes:
            print(f'\n  Mode: {mode.upper()}')
            if mode == 'zeroshot':
                res = run_zeroshot(dataset, subjects)
            elif mode == 'intra':
                res = run_intra(dataset, subjects,
                                train_frac=args.train_frac,
                                epochs=args.epochs)
            elif mode == 'pooled':
                res = run_pooled_intra(dataset, subjects,
                                       train_frac=args.train_frac,
                                       epochs=args.epochs)
            elif mode == 'ridge':
                res = run_ridge_baseline(dataset, subjects,
                                         train_frac=args.train_frac)
            else:
                res = run_finetune(
                    dataset, subjects,
                    train_frac=args.train_frac,
                    epochs=args.epochs,
                    leave_one_out=args.leave_one_out,
                )

            if res is not None:
                results[dataset][mode] = res

            # Save after every dataset × mode
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f'  Saved intermediate results -> {args.out}', flush=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f'\n\n{"="*80}')
    print('CROSS-DATASET SUMMARY (Pearson R, higher is better)')
    print(f'{"="*80}')
    hdr = f"{'Dataset':<14} {'Mode':<12}"
    for d in ROI_DISPLAY:
        hdr += f'{d[:8]:>10}'
    hdr += f'{"Avg.R":>10}'
    if any('avg_crps' in results.get(ds, {}).get(m, {})
           for ds in results for m in results[ds]):
        hdr += f'{"Avg.CRPS":>10}'
    print(hdr)
    print('─' * len(hdr))

    for dataset in datasets:
        for mode in modes:
            entry = results.get(dataset, {}).get(mode)
            if entry is None:
                continue
            per_roi = entry.get('per_roi', {})
            avg_r   = entry.get('avg_r', float('nan'))
            n_info  = entry.get('n_subjects', entry.get('n_folds', ''))
            row = f'{dataset:<14} {mode:<12}'
            for d in ROI_DISPLAY:
                r_val = per_roi.get(d, {})
                r = r_val.get('R', r_val.get('R_mean', float('nan')))
                row += f'{r:>10.3f}'
            row += f'{avg_r:>10.3f}'
            if n_info:
                row += f'  (n={n_info})'
            if 'avg_crps' in entry:
                row += f'  CRPS={entry["avg_crps"]:.4f}'
            print(row)

    print(f'{"="*80}')
    print(f'\nFull results saved to: {args.out}')


if __name__ == '__main__':
    main()
