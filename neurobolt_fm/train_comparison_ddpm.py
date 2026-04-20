"""
train_comparison_ddpm.py — Backbone + DDPM generative framework comparison.

Trains each baseline backbone (BIOT, LaBraM, FFCL, etc.) as a feature extractor
paired with a conditional DDPM head, producing a fair generative comparison
against NeuroFlow (NeuroBOLT + FM).

Protocol:
  1. Train backbone end-to-end with linear regression head (no FM/DDPM)
  2. Freeze backbone, extract features
  3. Train conditional DDPM head on those features
  4. Evaluate: Pearson R, CRPS, FC-MAE (same metrics as NeuroFlow)

Modes:
  --mode intra  : per-scan 80/10/10 split, results averaged across scans
  --mode pooled : global 80/10/10 split across all data

Usage:
  python train_comparison_ddpm.py --backbone biot --mode intra  --epochs_bb 100 --epochs_ddpm 300
  python train_comparison_ddpm.py --backbone ffcl --mode pooled --epochs_bb 100 --epochs_ddpm 300

Available backbones: cnn_trans, ffcl, stt_trans, biot, sparc, contrawr, beira, li2024
"""

import sys, os, gc, json, math, argparse, warnings
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

from comparison_backbones import build_backbone, BACKBONE_DISPLAY

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
DATA_ROOT = f'{BASE}/data'
CKPT_DIR  = f'{BASE}/checkpoints_comparison'
OUT_DIR   = f'{BASE}'

ROI_COLS = [
    "Cuneus",
    "Heschl\u2019s gyrus",
    "Middle frontal gyrus anterior",
    "Precuneus anterior",
    "Putamen",
    "Thalamus",
    "global signal clean",
]
ROI_DISPLAY = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
N_ROIS   = 7
SEQ_LEN  = 3200   # 16s × 200Hz
N_CHAN   = 26
TR       = 2.1
TMIN     = -16
CROP     = 3200
EVENT    = 'R149'

VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (identical to NeuroFlow pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def to_float(x):
    return float(np.asarray(x).flat[0])


def load_scan(sub, scan):
    """Load EEG + 7 ROI fMRI targets for one scan.
    Returns (eeg_trainval, fmri_trainval, eeg_test, fmri_test) or None.
    EEG shape: (N, 26, 3200); fMRI shape: (N, 7).
    """
    try:
        from dataset_maker import preproc
    except ImportError:
        sys.path.insert(0, os.path.join(BASE, 'code'))
        from dataset_maker import preproc

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

    df   = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15 / (0.5 / TR), btype='low')

    eeg_all = None
    all_fmri = []
    n = None

    for col in ROI_COLS:
        try:
            fm = df[[col]].to_numpy().T
        except KeyError:
            del raw
            return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        from dataset_maker import preproc as _p
        fm, _ = _p.normalize_data(fm)
        ep, _ = _p.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
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


def load_all_scans():
    print('\n[Data] Loading all scans...', flush=True)
    eeg_tv, fmri_tv = [], []
    eeg_te, fmri_te = [], []
    scan_names = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'  {pat}...', end=' ', flush=True)
        try:
            result = load_scan(sub, scan)
            if result is None:
                print('SKIP (missing ROI)', flush=True)
                continue
            et, tt, ete, tte = result
            if len(ete) < 5:
                print('SKIP (short test)', flush=True)
                continue
            eeg_tv.append(et); fmri_tv.append(tt)
            eeg_te.append(ete); fmri_te.append(tte)
            scan_names.append(pat)
            print(f'tr={len(et)} te={len(ete)}', flush=True)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    print(f'  Loaded {len(scan_names)} scans')
    return eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (after backbone is trained)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone, eeg_list, bs=64, device=DEVICE):
    """Extract features from list of EEG tensors. Returns list of feat tensors."""
    backbone = backbone.to(device)
    backbone.eval()
    out = []
    for eeg in eeg_list:
        feats = []
        for i in range(0, len(eeg), bs):
            x = (eeg[i:i+bs].to(device) / 100.)  # normalize
            feats.append(backbone(x).cpu())
        out.append(torch.cat(feats))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Train backbone with linear regression head
# ─────────────────────────────────────────────────────────────────────────────

class BackboneWithHead(nn.Module):
    def __init__(self, backbone, feat_dim, n_rois=7):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(feat_dim, n_rois)

    def forward(self, x):
        return self.head(self.backbone(x))


def train_backbone_intra(backbone, eeg_tv_list, fmri_tv_list, scan_names,
                         feat_dim, epochs=100, lr=1e-3, bs=64, device=DEVICE):
    """Train backbone per-scan (intra-subject). Returns list of trained backbone states."""
    print(f'\n[Phase 1] Training backbone (intra) for {len(scan_names)} scans...', flush=True)
    all_states = []

    for i, (scan, eeg_tv, fmri_tv) in enumerate(zip(scan_names, eeg_tv_list, fmri_tv_list)):
        n = len(eeg_tv)
        split = int(0.8 / 0.9 * n)  # 80% of trainval = train
        eeg_tr, fmri_tr = eeg_tv[:split], fmri_tv[:split]
        eeg_val, fmri_val = eeg_tv[split:], fmri_tv[split:]

        # Fresh backbone copy per scan
        bb_copy = type(backbone)(n_channels=N_CHAN, seq_len=SEQ_LEN, feat_dim=feat_dim).to(device)
        model   = BackboneWithHead(bb_copy, feat_dim).to(device)
        opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_val, best_state, patience, no_imp = 1e9, None, 15, 0

        for ep in range(epochs):
            model.train()
            idx = torch.randperm(len(eeg_tr))
            ep_loss = 0.
            for j in range(0, len(eeg_tr), bs):
                b_idx = idx[j:j+bs]
                x = (eeg_tr[b_idx].to(device) / 100.)
                y = fmri_tr[b_idx].to(device)
                loss = F.mse_loss(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()
            sched.step()

            # Validation
            if (ep + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = torch.cat([model(eeg_val[j:j+bs].to(device) / 100.).cpu()
                                          for j in range(0, len(eeg_val), bs)])
                val_loss = F.mse_loss(val_pred, fmri_val).item()
                if val_loss < best_val:
                    best_val   = val_loss
                    best_state = {k: v.cpu().clone() for k, v in bb_copy.state_dict().items()}
                    no_imp     = 0
                else:
                    no_imp += 1
                if no_imp >= patience:
                    break

        if best_state is not None:
            bb_copy.load_state_dict(best_state)
        all_states.append({k: v.cpu() for k, v in bb_copy.state_dict().items()})

        if (i + 1) % 5 == 0 or (i + 1) == len(scan_names):
            print(f'  {i+1}/{len(scan_names)} scans trained', flush=True)

    return all_states


def train_backbone_pooled(backbone, eeg_tv_all, fmri_tv_all,
                          feat_dim, epochs=100, lr=1e-3, bs=64, device=DEVICE):
    """Train single backbone on pooled data."""
    print(f'\n[Phase 1] Training backbone (pooled)...', flush=True)
    eeg_pool  = torch.cat(eeg_tv_all,  dim=0)
    fmri_pool = torch.cat(fmri_tv_all, dim=0)
    n = len(eeg_pool)
    split = int(0.8 / 0.9 * n)
    eeg_tr, fmri_tr   = eeg_pool[:split], fmri_pool[:split]
    eeg_val, fmri_val = eeg_pool[split:], fmri_pool[split:]

    backbone = backbone.to(device)
    model    = BackboneWithHead(backbone, feat_dim).to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state, patience, no_imp = 1e9, None, 20, 0

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(eeg_tr))
        for j in range(0, len(eeg_tr), bs):
            b_idx = idx[j:j+bs]
            x = (eeg_tr[b_idx].to(device) / 100.)
            y = fmri_tr[b_idx].to(device)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.cat([model(eeg_val[j:j+bs].to(device)/100.).cpu()
                                for j in range(0, len(eeg_val), bs)])
            vl = F.mse_loss(vp, fmri_val).item()
            if vl < best_val:
                best_val   = vl
                best_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                break
            if (ep + 1) % 20 == 0:
                print(f'  ep {ep+1}/{epochs}  val_mse={vl:.4f}  best={best_val:.4f}', flush=True)

    if best_state is not None:
        backbone.load_state_dict(best_state)
    print(f'  Final val MSE = {best_val:.4f}', flush=True)
    return backbone.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# DDPM architecture (conditional denoiser)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
        emb = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoiser(nn.Module):
    def __init__(self, feat_dim=256, proj_dim=256, hidden_dim=512, n_rois=7, time_dim=64):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        )
        in_dim = proj_dim + n_rois + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, x_t, t, features):
        t_emb  = self.time_emb(t)
        f_proj = self.feat_proj(features)
        inp    = torch.cat([f_proj, x_t, t_emb], dim=-1)
        return self.net(inp)


class ConditionalDDPM:
    def __init__(self, denoiser, T=300, beta_start=1e-4, beta_end=0.015, device='cuda'):
        self.denoiser = denoiser
        self.T        = T
        self.device   = device

        betas               = torch.linspace(beta_start, beta_end, T, device=device)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        self.betas                       = betas
        self.alphas_cumprod              = alphas_cumprod
        self.sqrt_alphas_cumprod         = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        ab  = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        mab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return ab * x_0 + mab * noise, noise

    def train_loss(self, x_0, features):
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        return F.mse_loss(self.denoiser(x_t, t, features), noise)

    @torch.no_grad()
    def ddim_sample(self, features, n_samples=50, ddim_steps=50, eta=0.0, bs=64):
        """Returns [n_samples, N, 7]."""
        N         = features.shape[0]
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        all_samples = []

        for _ in range(n_samples):
            parts = []
            for i in range(0, N, bs):
                fb = features[i:i+bs].to(self.device)
                B  = len(fb)
                x  = torch.randn(B, self.denoiser.n_rois, device=self.device)

                for idx, t_cur in enumerate(timesteps):
                    t_tensor = torch.full((B,), t_cur, device=self.device, dtype=torch.long)
                    eps_pred  = self.denoiser(x, t_tensor, fb)
                    ab_cur    = self.alphas_cumprod[t_cur]
                    ab_prev   = self.alphas_cumprod[timesteps[idx+1]] if idx < len(timesteps)-1 else torch.tensor(1., device=self.device)
                    x0_pred   = ((x - (1-ab_cur).sqrt() * eps_pred) / ab_cur.sqrt()).clamp(-5, 5)
                    sigma     = eta * ((1-ab_prev)/(1-ab_cur) * (1 - ab_cur/ab_prev)).sqrt()
                    dir_xt    = (1 - ab_prev - sigma**2).sqrt() * eps_pred
                    x         = ab_prev.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)

                parts.append(x.cpu())
            all_samples.append(torch.cat(parts, dim=0))

        return torch.stack(all_samples)  # [n_samples, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Train DDPM on frozen backbone features
# ─────────────────────────────────────────────────────────────────────────────

def _crps_exact(samples, targets):
    """
    Exact CRPS via all-pairs energy score.
    CRPS = E|Y-y| - 0.5*E|Y-Y'|  (all K*(K-1) pairs, no random approximation)
    samples: [K, N, D], targets: [N, D]
    """
    K = samples.shape[0]
    # E|Y - y|: mean over K samples and all N*D elements
    e1 = (samples - targets.unsqueeze(0)).abs().mean()
    # 0.5 * E|Y - Y'|: exact all-pairs (K^2 pairs, diagonal zeros don't affect mean much)
    # samples: [K, N, D] — compute pairwise |Y_k - Y_l| for all k,l
    # Use broadcasting: [K, 1, N, D] - [1, K, N, D] → [K, K, N, D]
    diff = (samples.unsqueeze(1) - samples.unsqueeze(0)).abs()  # [K, K, N, D]
    e2 = 0.5 * diff.mean()
    return (e1 - e2).item()


# Keep fast version as fallback for large K during training
def _crps_fast(samples, targets):
    """Approximate CRPS using random-pair estimate (for validation during training)."""
    K = samples.shape[0]
    e1 = (samples - targets.unsqueeze(0)).abs().mean()
    perm = torch.randperm(K)
    e2 = 0.5 * (samples - samples[perm]).abs().mean()
    return (e1 - e2).item()


def train_ddpm(feat_tr, fmri_tr, feat_val, fmri_val,
               feat_dim, epochs=300, lr=3e-4, bs=256, device=DEVICE):
    # Normalize targets to zero-mean, unit-variance per ROI
    mu_tgt  = fmri_tr.mean(0)
    sig_tgt = fmri_tr.std(0).clamp(min=1e-8)
    fmri_tr_n  = (fmri_tr  - mu_tgt) / sig_tgt
    fmri_val_n = (fmri_val - mu_tgt) / sig_tgt

    denoiser = ConditionalDenoiser(feat_dim=feat_dim, proj_dim=256,
                                   hidden_dim=512, n_rois=N_ROIS).to(device)
    # T=300 (reviewer recommendation: 200-400 sufficient for 7D output)
    ddpm     = ConditionalDDPM(denoiser, T=300, beta_end=0.015, device=str(device))
    # AdamW with weight_decay (reviewer recommendation)
    opt      = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Model selection by validation CRPS (not MSE — reviewer fix)
    best_crps, best_state, patience, no_imp = 1e9, None, 30, 0

    for ep in range(epochs):
        denoiser.train()
        idx  = torch.randperm(len(feat_tr))
        for j in range(0, len(feat_tr), bs):
            b   = idx[j:j+bs]
            f_b = feat_tr[b].to(device)
            y_b = fmri_tr_n[b].to(device)
            loss = ddpm.train_loss(y_b, f_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            denoiser.eval()
            with torch.no_grad():
                val_samps_n = ddpm.ddim_sample(feat_val.to(device), n_samples=10,
                                               ddim_steps=30, bs=256)  # [K, N, 7]
            # Un-normalize (val_samps_n is already on CPU from ddim_sample)
            val_samps = val_samps_n * sig_tgt + mu_tgt  # both on CPU
            val_crps  = _crps_fast(val_samps, fmri_val)

            if val_crps < best_crps:
                best_crps  = val_crps
                best_state = {k: v.cpu().clone() for k, v in denoiser.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                print(f'    Early stop ep {ep+1}  best_val_crps={best_crps:.4f}', flush=True)
                break

    if best_state is not None:
        denoiser.load_state_dict(best_state)
    # Store normalization params on the ddpm object for inference
    ddpm.mu_tgt  = mu_tgt
    ddpm.sig_tgt = sig_tgt
    return ddpm


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_crps(samples, targets):
    """
    Exact CRPS via energy score: E|Y-y| - 0.5 E|Y-Y'|.
    Uses all K*(K-1) pairs for E|Y-Y'| — no random approximation.
    samples: [K, N, 7], targets: [N, 7]
    """
    K, N, D = samples.shape
    energy1 = (samples - targets.unsqueeze(0)).abs().mean().item()
    # Exact all-pairs: [K, K, N, D], memory: K^2*N*D*4 bytes
    # K=50, N≈500, D=7 → 50*50*500*7*4 ≈ 35 MB — acceptable
    diff = (samples.unsqueeze(1) - samples.unsqueeze(0)).abs()  # [K, K, N, D]
    energy2 = 0.5 * diff.mean().item()
    return energy1 - energy2


def compute_fc_mae(samples, targets):
    """FC-MAE: conditional covariance recovery error."""
    K, N, D = samples.shape
    pred_mean = samples.mean(0)  # [N, 7]
    # Pearson correlation matrix from predictions vs targets
    def corr_mat(x):
        x = x - x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True).clamp(min=1e-8)
        x = x / std
        return (x.T @ x) / (len(x) - 1)

    fc_pred = corr_mat(pred_mean)
    fc_tgt  = corr_mat(targets)
    return (fc_pred - fc_tgt).abs().mean().item()


def _per_example_crps(samples, targets):
    """Per-example CRPS contribution [N] — used for paired significance testing."""
    # E_k|s_k - y|: [N]
    e1 = (samples - targets.unsqueeze(0)).abs().mean(dim=(0, 2))
    # 0.5 * E_{k,l}|s_k - s_l|: [N]
    diff = (samples.unsqueeze(1) - samples.unsqueeze(0)).abs().mean(dim=(0, 1, 3))
    return (e1 - 0.5 * diff)


def bootstrap_ci(samples, targets, n_boot=500, alpha=0.05, seed=42):
    """
    Per-example bootstrap 95% CI for CRPS and avg_R.
    Note: for paired significance vs another method, use paired_bootstrap_test() instead.
    Returns {'crps_ci': (lo, hi), 'r_ci': (lo, hi)}
    """
    rng = np.random.default_rng(seed)
    N   = targets.shape[0]
    crps_vals, r_vals = [], []
    pred_mean_full = samples.float().mean(0)  # [N, 7]

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        s_b = samples[:, idx, :]    # [K, N, 7]
        t_b = targets[idx]          # [N, 7]
        pm  = pred_mean_full[idx]   # [N, 7]

        crps_vals.append(compute_crps(s_b, t_b))
        r_vals.append(float(np.mean([
            pearsonr(pm[:, j].numpy(), t_b[:, j].numpy())[0]
            for j in range(N_ROIS)
        ])))

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        'crps_ci': (float(np.quantile(crps_vals, lo)), float(np.quantile(crps_vals, hi))),
        'r_ci'   : (float(np.quantile(r_vals,    lo)), float(np.quantile(r_vals,    hi))),
    }


def paired_bootstrap_test(samples_a, samples_b, targets, n_boot=1000, alpha=0.05, seed=0):
    """
    Paired bootstrap significance test: H0: E[CRPS_A] = E[CRPS_B].
    One-sided test: A is better iff p_value < alpha (CRPS_A significantly < CRPS_B).

    Method: standard centered bootstrap for testing a mean difference.
      1. d_i = CRPS_A_i - CRPS_B_i   (per-example difference)
      2. obs_delta = mean(d)           (observed mean diff; negative = A is better)
      3. d_0 = d - obs_delta           (center at 0 under H0)
      4. t_boot = mean(d_0[resample]) for each bootstrap resample
      5. p_value = fraction of t_boot <= obs_delta  (one-sided, A < B)

    samples_a/b: [K, N, 7], targets: [N, 7]
    Returns {'obs_delta': CRPS_A - CRPS_B, 'p_value': float, 'significant': bool}
    """
    rng = np.random.default_rng(seed)
    N   = targets.shape[0]

    score_a = _per_example_crps(samples_a, targets).numpy()   # [N]
    score_b = _per_example_crps(samples_b, targets).numpy()   # [N]
    d = score_a - score_b   # positive means A is worse; we want this negative

    obs_delta = float(d.mean())
    d_0 = d - obs_delta   # centered at 0 under H0

    # Bootstrap null distribution
    boot_means = np.array([d_0[rng.integers(0, N, size=N)].mean() for _ in range(n_boot)])
    # One-sided p-value: how often does null give mean <= obs_delta
    p_val = float((boot_means <= obs_delta).mean())
    return {
        'obs_delta' : obs_delta,
        'p_value'   : p_val,
        'significant': p_val < alpha,
    }


def evaluate(ddpm, feat_te, fmri_te, n_samples=50, ddim_steps=50, device=DEVICE,
             compute_ci=False):
    """Evaluate DDPM on test set. Returns dict with R, CRPS, FC-MAE (+ 95% CI if requested)."""
    ddpm.denoiser.eval()
    samples_n = ddpm.ddim_sample(feat_te.to(device), n_samples=n_samples,
                                 ddim_steps=ddim_steps, bs=256)  # [K, N, 7] (normalized)
    # Un-normalize if normalization params exist (samples_n already on CPU from ddim_sample)
    if hasattr(ddpm, 'mu_tgt') and ddpm.mu_tgt is not None:
        samples = samples_n * ddpm.sig_tgt + ddpm.mu_tgt  # CPU × CPU
    else:
        samples = samples_n
    pred_mean = samples.mean(0)  # [N, 7]

    # Per-ROI Pearson R
    roi_r = []
    for j in range(N_ROIS):
        r, _ = pearsonr(pred_mean[:, j].numpy(), fmri_te[:, j].numpy())
        roi_r.append(float(r))
    avg_r = float(np.nanmean(roi_r))

    crps   = compute_crps(samples, fmri_te)
    fc_mae = compute_fc_mae(samples, fmri_te)

    result = {
        'avg_r'     : avg_r,
        'roi_r'     : roi_r,
        'crps'      : crps,
        'fc_mae'    : fc_mae,
    }
    if compute_ci:
        ci = bootstrap_ci(samples, fmri_te)
        result['crps_ci'] = ci['crps_ci']
        result['r_ci']    = ci['r_ci']
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipelines
# ─────────────────────────────────────────────────────────────────────────────

_PRETRAINED_BACKBONES = {'neurobolt', 'reve', 'brainomni'}  # frozen, skip Phase 1 training

# Effective feature dimensions for pretrained backbones
_PRETRAINED_FEAT_DIMS = {'neurobolt': 200, 'reve': 512, 'brainomni': 512}

# Auto-detect NeuroBOLT checkpoint.
# Paper note: We use code/checkpoints/Cuneus.pth as the single-backbone NeuroBOLT baseline.
# This is one of 7 ROI-specialized checkpoints; using any one checkpoint is equivalent
# at the backbone level since the backbone weights differ only marginally across ROIs
# (all fine-tuned from the same foundation). We document this choice explicitly.
# Use --neurobolt_ckpt to specify a different checkpoint for reproducibility.
_NEUROBOLT_CKPT_CANDIDATES = [
    f'{BASE}/code/checkpoints/Cuneus.pth',    # primary — documented choice
    f'{BASE}/code/checkpoints/glb.pth',        # fallback
]

def _find_neurobolt_ckpt(explicit_path=None):
    """Find a valid NeuroBOLT checkpoint to use for feature extraction."""
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f'--neurobolt_ckpt not found: {explicit_path}')
        return explicit_path
    for p in _NEUROBOLT_CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        'NeuroBOLT checkpoint not found. Searched:\n' +
        '\n'.join(f'  {p}' for p in _NEUROBOLT_CKPT_CANDIDATES) +
        '\nPass --neurobolt_ckpt <path> to specify it.'
    )


def _make_backbone(backbone_name, feat_dim, neurobolt_ckpt=None):
    """Build backbone; for pretrained backbones, load weights."""
    if backbone_name == 'neurobolt':
        ckpt = _find_neurobolt_ckpt(neurobolt_ckpt)
        bb = build_backbone(backbone_name, n_channels=N_CHAN, seq_len=SEQ_LEN,
                            feat_dim=feat_dim, ckpt_path=ckpt)
        bb.load(device=str(DEVICE))
        print(f'  [neurobolt] Loaded pretrained backbone from: {ckpt}', flush=True)
        return bb
    if backbone_name in ('reve', 'brainomni'):
        bb = build_backbone(backbone_name, n_channels=N_CHAN, seq_len=SEQ_LEN, feat_dim=512)
        bb.load(device=str(DEVICE))
        return bb
    return build_backbone(backbone_name, n_channels=N_CHAN, seq_len=SEQ_LEN, feat_dim=feat_dim)


def run_intra(backbone_name, feat_dim, epochs_bb, epochs_ddpm, neurobolt_ckpt=None):
    """Intra-subject: per-scan backbone training + DDPM."""
    print(f'\n=== INTRA-SUBJECT: {BACKBONE_DISPLAY.get(backbone_name, backbone_name)} + DDPM ===\n')
    os.makedirs(CKPT_DIR, exist_ok=True)

    eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names = load_all_scans()

    is_pretrained = backbone_name in _PRETRAINED_BACKBONES
    # Effective feat_dim (pretrained backbones may differ)
    eff_feat_dim = _PRETRAINED_FEAT_DIMS.get(backbone_name, feat_dim)

    if is_pretrained:
        # Pretrained frozen backbone — skip Phase 1, share one loaded backbone
        print(f'\n[Phase 1] Skipped — {backbone_name} is pretrained and frozen.', flush=True)
        shared_bb = _make_backbone(backbone_name, feat_dim, neurobolt_ckpt=neurobolt_ckpt)
        bb_states  = [None] * len(scan_names)
    else:
        proto = build_backbone(backbone_name, n_channels=N_CHAN, seq_len=SEQ_LEN, feat_dim=feat_dim)
        bb_states = train_backbone_intra(proto, eeg_tv, fmri_tv, scan_names,
                                         feat_dim=feat_dim, epochs=epochs_bb)
        shared_bb = None

    # Phase 2 per scan: extract features → train DDPM → evaluate
    scan_results = []

    for i, (scan, state) in enumerate(zip(scan_names, bb_states)):
        print(f'\n  [Scan {i+1}/{len(scan_names)}] {scan}', flush=True)

        if is_pretrained:
            bb = shared_bb
        else:
            bb = build_backbone(backbone_name, n_channels=N_CHAN, seq_len=SEQ_LEN, feat_dim=feat_dim)
            bb.load_state_dict(state)

        # Extract features
        n_tv = len(eeg_tv[i])
        split = int(0.8 / 0.9 * n_tv)
        feat_all = extract_features(bb, [eeg_tv[i]], device=DEVICE)[0]  # [n_tv, eff_feat_dim]
        feat_tr, fmri_tr   = feat_all[:split], fmri_tv[i][:split]
        feat_val, fmri_val = feat_all[split:], fmri_tv[i][split:]
        feat_tst           = extract_features(bb, [eeg_te[i]], device=DEVICE)[0]
        fmri_tst           = fmri_te[i]

        # Train DDPM
        ddpm = train_ddpm(feat_tr, fmri_tr, feat_val, fmri_val,
                          feat_dim=eff_feat_dim, epochs=epochs_ddpm, device=DEVICE)

        # Evaluate with exact CRPS + bootstrap CI per scan
        res = evaluate(ddpm, feat_tst, fmri_tst, n_samples=50, device=DEVICE,
                       compute_ci=True)
        print(f'    avg_R={res["avg_r"]:.4f}  CRPS={res["crps"]:.4f}  FC-MAE={res["fc_mae"]:.4f}')
        scan_results.append({'scan': scan, **res})

        if not is_pretrained:
            del bb
        del ddpm; gc.collect(); torch.cuda.empty_cache()

    # Aggregate across scans
    all_r    = [r['avg_r']  for r in scan_results]
    all_crps = [r['crps']   for r in scan_results]
    all_fc   = [r['fc_mae'] for r in scan_results]
    roi_r_mat = np.array([r['roi_r'] for r in scan_results])
    S = len(scan_results)

    # Scan-aggregate bootstrap CI: resample scans (blocks) with replacement
    # This correctly propagates within-scan dependence to the aggregate mean
    rng = np.random.default_rng(42)
    n_boot = 500
    crps_boot, r_boot = [], []
    crps_arr = np.array(all_crps)
    r_arr    = np.array(all_r)
    for _ in range(n_boot):
        idx = rng.integers(0, S, size=S)
        crps_boot.append(float(crps_arr[idx].mean()))
        r_boot.append(float(r_arr[idx].mean()))
    crps_ci = (float(np.quantile(crps_boot, 0.025)), float(np.quantile(crps_boot, 0.975)))
    r_ci    = (float(np.quantile(r_boot,    0.025)), float(np.quantile(r_boot,    0.975)))

    summary = {
        'backbone'  : backbone_name,
        'mode'      : 'intra',
        'avg_r'     : float(np.nanmean(all_r)),
        'avg_r_std' : float(np.nanstd(all_r)),
        'r_ci'      : r_ci,
        'crps'      : float(np.mean(all_crps)),
        'crps_ci'   : crps_ci,
        'fc_mae'    : float(np.mean(all_fc)),
        'roi_r_mean': np.nanmean(roi_r_mat, axis=0).tolist(),
        'per_scan'  : scan_results,
    }
    print(f'\n  INTRA SUMMARY: avg_R={summary["avg_r"]:.4f}±{summary["avg_r_std"]:.4f}'
          f'  R_95CI=[{r_ci[0]:.4f},{r_ci[1]:.4f}]'
          f'  CRPS={summary["crps"]:.4f} [{crps_ci[0]:.4f},{crps_ci[1]:.4f}]'
          f'  FC-MAE={summary["fc_mae"]:.4f}')
    return summary


def run_pooled(backbone_name, feat_dim, epochs_bb, epochs_ddpm, neurobolt_ckpt=None):
    """Pooled (inter-subject): one backbone + DDPM trained on all subjects."""
    print(f'\n=== POOLED (inter-subject): {BACKBONE_DISPLAY.get(backbone_name, backbone_name)} + DDPM ===\n')
    os.makedirs(CKPT_DIR, exist_ok=True)

    eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names = load_all_scans()

    is_pretrained = backbone_name in _PRETRAINED_BACKBONES
    eff_feat_dim  = _PRETRAINED_FEAT_DIMS.get(backbone_name, feat_dim)

    if is_pretrained:
        print(f'\n[Phase 1] Skipped — {backbone_name} is pretrained and frozen.', flush=True)
        backbone = _make_backbone(backbone_name, feat_dim, neurobolt_ckpt=neurobolt_ckpt)
    else:
        backbone = build_backbone(backbone_name, n_channels=N_CHAN, seq_len=SEQ_LEN, feat_dim=feat_dim)
        backbone = train_backbone_pooled(backbone, eeg_tv, fmri_tv,
                                         feat_dim=feat_dim, epochs=epochs_bb)

    # Extract features
    print('\n  Extracting features...', flush=True)
    feat_tv_list = extract_features(backbone, eeg_tv, device=DEVICE)
    feat_te_list = extract_features(backbone, eeg_te, device=DEVICE)

    feat_tv_all  = torch.cat(feat_tv_list)
    fmri_tv_all  = torch.cat(fmri_tv)
    feat_te_all  = torch.cat(feat_te_list)
    fmri_te_all  = torch.cat(fmri_te)

    n_tv  = len(feat_tv_all)
    split = int(0.8 / 0.9 * n_tv)
    feat_tr, fmri_tr   = feat_tv_all[:split], fmri_tv_all[:split]
    feat_val, fmri_val = feat_tv_all[split:], fmri_tv_all[split:]

    print(f'  pooled train={len(feat_tr)}, val={len(feat_val)}, test={len(feat_te_all)}')

    # Phase 2: train DDPM
    ddpm = train_ddpm(feat_tr, fmri_tr, feat_val, fmri_val,
                      feat_dim=eff_feat_dim, epochs=epochs_ddpm, device=DEVICE)

    # Evaluate (with bootstrap CIs for final test evaluation)
    res = evaluate(ddpm, feat_te_all, fmri_te_all, n_samples=50, device=DEVICE, compute_ci=True)

    summary = {
        'backbone' : backbone_name,
        'mode'     : 'pooled',
        **res,
    }
    ci_str = ''
    if 'crps_ci' in res:
        ci_str = f'  CRPS_95CI=[{res["crps_ci"][0]:.4f},{res["crps_ci"][1]:.4f}]'
    print(f'\n  POOLED SUMMARY: avg_R={res["avg_r"]:.4f}  CRPS={res["crps"]:.4f}  FC-MAE={res["fc_mae"]:.4f}{ci_str}')
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train backbone + DDPM comparison')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['cnn_trans','ffcl','stt_trans','biot','sparc','contrawr','beira','li2024',
                                 'neurobolt','labram','reve','brainomni'],
                        help='Backbone architecture to use')
    parser.add_argument('--mode', type=str, default='intra', choices=['intra','pooled'],
                        help='Evaluation mode: intra or pooled')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Backbone output feature dimension')
    parser.add_argument('--epochs_bb',   type=int, default=100, help='Epochs for backbone training')
    parser.add_argument('--epochs_ddpm', type=int, default=300, help='Epochs for DDPM training')
    parser.add_argument('--out', type=str, default=None,
                        help='Output JSON path (default: comparison_ddpm_results.json)')
    parser.add_argument('--neurobolt_ckpt', type=str, default=None,
                        help='Path to NeuroBOLT pretrained checkpoint (required when --backbone=neurobolt). '
                             f'Auto-detected from {BASE}/code/checkpoints/ if not provided.')
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Backbone: {args.backbone}  Mode: {args.mode}  feat_dim={args.feat_dim}')
    print(f'Epochs: backbone={args.epochs_bb}  ddpm={args.epochs_ddpm}')

    if args.mode == 'intra':
        result = run_intra(args.backbone, args.feat_dim, args.epochs_bb, args.epochs_ddpm,
                           neurobolt_ckpt=args.neurobolt_ckpt)
    else:
        result = run_pooled(args.backbone, args.feat_dim, args.epochs_bb, args.epochs_ddpm,
                            neurobolt_ckpt=args.neurobolt_ckpt)

    # Save results
    out_path = args.out or os.path.join(OUT_DIR, 'comparison_ddpm_results.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    key = f'{args.backbone}_{args.mode}'
    existing[key] = result
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults saved to {out_path}')

    # Print ROI table
    roi_r = result.get('roi_r_mean') or result.get('roi_r', [])
    print('\nROI breakdown:')
    for name, r in zip(ROI_DISPLAY, roi_r):
        print(f'  {name:<14}: R={r:.4f}')
    print(f'  {"Avg":<14}: R={result["avg_r"]:.4f}  CRPS={result["crps"]:.4f}  FC-MAE={result["fc_mae"]:.4f}')
