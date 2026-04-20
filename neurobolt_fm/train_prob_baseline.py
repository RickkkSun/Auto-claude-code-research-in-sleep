"""
FM-NeuroBOLT Probabilistic Joint Baseline — Round 3 fix

Implements:
1. Joint Multivariate Gaussian head (full Cholesky covariance) on 1400D 7-backbone features
   - Trained with exact multivariate Gaussian NLL loss
   - Full cross-ROI covariance modeling
   - Provides: CRPS, 90% coverage, NLL, cross-ROI Cov MAE (proper probabilistic baseline)

2. FM v3b temperature scaling calibration
   - Grid-search temperature T on validation set
   - Scale FM ODE initial noise: x_0 ~ N(0, T^2 * I)
   - Achieves better calibration while preserving joint structure

Key question: Does FM v3b outperform matched Gaussian head on:
  (a) Covariance recovery (cross-ROI MAE)?
  (b) CRPS?
  (c) Coverage?
If yes → FM's generative advantage over Gaussian parametric family is demonstrated.
"""

import sys, os, gc, json, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
mne.set_log_level('WARNING')
import pandas as pd
from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, norm as scipy_norm
from timm.models import create_model
import models.model
from dataset_maker import preproc

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
V3B_CKPT = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
OUT_JSON  = f'{BASE}/prob_baseline_results.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Re-use feature extraction from v3b script
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, BASE)
from train_fm_v3b_multisource import (
    load_scan_all_rois, extract_features as exfeat, build_backbone,
    ROI_COLS, ROIS as ROIS_V3B, ALL_SCANS, DATA_ROOT, CKPT_DIR,
    MultiSourceFMHead
)

ROI_DISPLAY = ['Cuneus', "Heschl's Gyrus", 'Mid. Frontal', 'Precuneus Ant.',
               'Putamen', 'Thalamus', 'Global Signal']

# ─────────────────────────────────────────────────────────────────────────────
# Joint Multivariate Gaussian Head (full Cholesky covariance)
# ─────────────────────────────────────────────────────────────────────────────
class JointGaussianHead(nn.Module):
    """
    Maps 1400D features → 7D mean + lower-triangular Cholesky factor L (7x7).
    Σ = L @ L.T + eps*I  (positive definite by construction)
    Training loss: exact multivariate Gaussian NLL
    """
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7):
        super().__init__()
        self.n_rois = n_rois
        # Feature projector (same as FM head)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        )
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        # Mean head
        self.mu_head = nn.Linear(hidden_dim, n_rois)
        # Lower-triangular Cholesky entries: n_rois*(n_rois+1)/2 parameters
        n_chol = n_rois * (n_rois + 1) // 2  # 28 for 7 ROIs
        self.chol_head = nn.Linear(hidden_dim, n_chol)

    def forward(self, feat):
        h = self.feat_proj(feat)
        h = self.shared(h)
        mu = self.mu_head(h)            # (B, 7)
        chol_vec = self.chol_head(h)    # (B, 28)

        # Build lower-triangular Cholesky matrix
        B = feat.shape[0]
        L = torch.zeros(B, self.n_rois, self.n_rois, device=feat.device)
        idx = torch.tril_indices(self.n_rois, self.n_rois)
        L[:, idx[0], idx[1]] = chol_vec

        # Enforce positive diagonal (for valid Cholesky)
        diag_idx = torch.arange(self.n_rois)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-4

        return mu, L  # Σ = L @ L.T

    def nll_loss(self, feat, target_norm):
        """Negative log-likelihood of multivariate Gaussian."""
        mu, L = self.forward(feat)
        # torch.distributions.MultivariateNormal with scale_tril=L
        dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
        nll = -dist.log_prob(target_norm)  # (B,)
        return nll.mean()

    @torch.no_grad()
    def sample(self, feat, n_samples=50):
        """
        Generate n_samples from the Gaussian for each input.
        Returns (n_samples, N_test, 7) numpy array.
        """
        mu, L = self.forward(feat)      # mu: (N,7), L: (N,7,7)
        # Σ = L @ L.T
        Sigma = L @ L.transpose(-1, -2)  # (N, 7, 7)
        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mu)  # (N, 7)
            s = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)  # L @ eps ~ N(0, Sigma)
            samples.append(s.cpu().numpy())
        return np.stack(samples, axis=0)  # (n_samples, N_test, 7)

    @torch.no_grad()
    def predict_cov(self, feat):
        """Return mean predicted covariance matrix (N, 7, 7) → average → (7,7)."""
        _, L = self.forward(feat)
        Sigma = L @ L.transpose(-1, -2)
        return Sigma.cpu().numpy()  # (N, 7, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic metrics
# ─────────────────────────────────────────────────────────────────────────────
def crps_ensemble(samples, target):
    """CRPS. samples: (N_s, N_t). target: (N_t,)."""
    N, T = samples.shape
    term1 = np.mean(np.abs(samples - target[None, :]), axis=0)
    term2 = np.zeros(T)
    for i in range(N):
        for j in range(i+1, N):
            term2 += np.abs(samples[i] - samples[j])
    term2 = term2 / (N * (N-1) / 2)
    return float(np.mean(term1 - 0.5 * term2))

def interval_coverage(samples, target, alpha=0.1):
    lo = np.percentile(samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return float(np.mean((target >= lo) & (target <= hi)))

def nll_gaussian_samples(samples, target):
    mu = samples.mean(0); std = samples.std(0) + 1e-6
    return float(np.mean(-scipy_norm.logpdf(target, loc=mu, scale=std)))


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate Joint Gaussian Head
# ─────────────────────────────────────────────────────────────────────────────
def train_gaussian_head(feat_tr, tgt_tr_raw, feat_val, tgt_val_raw, epochs=300, lr=3e-4, bs=128):
    mu  = torch.tensor(tgt_tr_raw.mean(0).numpy(), dtype=torch.float32)
    sig = torch.tensor(tgt_tr_raw.std(0).numpy() + 1e-6, dtype=torch.float32)

    tgt_tr_norm  = (tgt_tr_raw  - mu) / sig
    tgt_val_norm = (tgt_val_raw - mu) / sig

    model = JointGaussianHead(feat_dim=feat_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(len(feat_tr) / bs))
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*steps)

    best_val_nll = float('inf')
    best_state = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            x = feat_tr[idx].to(DEVICE)
            y = tgt_tr_norm[idx].to(DEVICE)
            loss = model.nll_loss(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

        # Validation NLL
        if (ep + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_nll = model.nll_loss(feat_val.to(DEVICE), tgt_val_norm.to(DEVICE)).item()
            print(f'    ep {ep+1}/{epochs}  train_loss={loss.item():.4f}  val_nll={val_nll:.4f}')
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'    Loaded best model (val_nll={best_val_nll:.4f})')

    return model, mu, sig


def eval_gaussian_head(model, mu, sig, feat_test, tgt_test, n_samples=50):
    """Evaluate Gaussian head: R, CRPS, Coverage, NLL, Cov MAE."""
    model.eval()
    true = tgt_test.numpy()
    mu_np = mu.numpy(); sig_np = sig.numpy()

    with torch.no_grad():
        samples_norm = model.sample(feat_test.to(DEVICE), n_samples=n_samples)
        # samples_norm: (n_samples, N_test, 7) — in normalized space

    # De-normalize
    samples_denorm = samples_norm * sig_np[None, None, :] + mu_np[None, None, :]

    results = {}
    for j, disp in enumerate(ROI_DISPLAY):
        roi_samples_norm = samples_norm[:, :, j]    # (n_s, N_test) normalized
        roi_true_norm    = (true[:, j] - mu_np[j]) / sig_np[j]

        crps  = crps_ensemble(roi_samples_norm, roi_true_norm)
        cover = interval_coverage(roi_samples_norm, roi_true_norm, alpha=0.1)
        nll   = nll_gaussian_samples(roi_samples_norm, roi_true_norm)

        roi_pred_denorm = samples_denorm[:, :, j].mean(0)
        r_val, _ = pearsonr(roi_pred_denorm, true[:, j])

        results[disp] = {'R': float(r_val), 'CRPS': float(crps),
                         'Coverage_90': float(cover), 'NLL': float(nll)}
        print(f'  {disp:<20}: R={r_val:.3f}  CRPS={crps:.4f}  '
              f'Cover90={cover:.3f}  NLL={nll:.4f}')

    # Covariance recovery
    pred_mean_denorm = samples_denorm.mean(0)  # (N_test, 7)
    pred_corr = np.corrcoef(pred_mean_denorm.T)
    true_corr = np.corrcoef(true.T)
    off_diag  = ~np.eye(7, dtype=bool)
    cov_mae   = float(np.mean(np.abs(pred_corr[off_diag] - true_corr[off_diag])))
    print(f'  Cross-ROI Cov MAE: {cov_mae:.4f}')
    results['cov_mae'] = cov_mae
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FM v3b temperature scaling calibration
# ─────────────────────────────────────────────────────────────────────────────
def eval_fm_with_temperature(head, mu, sig, feat_test, tgt_test, temperature=1.0, n_samples=50):
    """Run FM v3b with scaled initial noise (temperature scaling)."""
    head.eval()
    N_test = len(feat_test)
    true = tgt_test.numpy()
    mu_np = np.array(mu); sig_np = np.array(sig)

    with torch.no_grad():
        feat_gpu = feat_test.to(DEVICE)
        all_samples = []
        for s in range(n_samples):
            # Temperature scaling: initial noise scaled by T
            x = torch.randn(N_test, 7, device=DEVICE) * temperature
            dt = 1.0 / 100
            for i in range(100):
                t = torch.full((N_test, 1), i / 100, device=DEVICE)
                x = x + head(feat_gpu, x, t) * dt
            all_samples.append(x.cpu().numpy())

    samples_np = np.stack(all_samples, axis=0)  # (n_samples, N_test, 7)

    # Eval per ROI
    coverages = []
    for j, disp in enumerate(ROI_DISPLAY):
        roi_samples = samples_np[:, :, j]
        roi_true    = (true[:, j] - mu_np[j]) / sig_np[j]
        cover = interval_coverage(roi_samples, roi_true, alpha=0.1)
        coverages.append(cover)

    return samples_np, float(np.mean(coverages))


def find_fm_temperature(head, mu, sig, feat_val, tgt_val):
    """Find temperature that maximizes 90% coverage on validation set."""
    best_T = 1.0
    best_dist = float('inf')
    print('\n  FM temperature calibration (target coverage=0.90):')
    for T in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        _, cover = eval_fm_with_temperature(head, mu, sig, feat_val, tgt_val, temperature=T, n_samples=30)
        dist = abs(cover - 0.90)
        print(f'    T={T:.1f}: coverage={cover:.3f}  dist_to_0.9={dist:.3f}')
        if dist < best_dist:
            best_dist = dist
            best_T = T
    print(f'  Best temperature: T={best_T:.1f} (coverage dist={best_dist:.3f})')
    return best_T


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '='*60)
    print('FM-NeuroBOLT: Probabilistic Joint Baseline + FM Calibration')
    print('='*60)
    print(f'Device: {DEVICE}')

    # ── Load features from 7 backbones ──────────────────────────────────────
    print('\n[1] Loading 7-backbone features (train/val/test)...')

    eeg_train_scans, eeg_val_scans, eeg_test_scans = [], [], []
    tgt_train_list,  tgt_val_list,  tgt_test_list  = [], [], []

    for (sub, scan) in ALL_SCANS:
        try:
            result = load_scan_all_rois(sub, scan)
            if result is None: continue
            et, tt, ete, tte = result
            if len(ete) < 5: continue
            # Carve out a validation chunk from training data (last 10% of train)
            n_tr = len(et)
            val_start = int(0.85 * n_tr)
            eeg_train_scans.append(et[:val_start])
            eeg_val_scans.append(et[val_start:])
            tgt_train_list.append(tt[:val_start])
            tgt_val_list.append(tt[val_start:])
            eeg_test_scans.append(ete)
            tgt_test_list.append(tte)
        except Exception:
            pass

    print(f'  Loaded {len(eeg_train_scans)} scans')

    # Extract features from all 7 backbones
    all_feat_train, all_feat_val, all_feat_test = [], [], []
    for roi_col, ckpt_fname, display in ROIS_V3B:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'  {display}...', end=' ', flush=True)
        backbone = build_backbone(ckpt_path)
        all_feat_train.append(torch.cat([exfeat(backbone, e) for e in eeg_train_scans], dim=0))
        all_feat_val.append(torch.cat([exfeat(backbone, e) for e in eeg_val_scans], dim=0))
        all_feat_test.append(torch.cat([exfeat(backbone, e) for e in eeg_test_scans], dim=0))
        del backbone; gc.collect(); torch.cuda.empty_cache()
        print('done')

    feat_train = torch.cat(all_feat_train, dim=1)  # (N_train, 1400)
    feat_val   = torch.cat(all_feat_val,   dim=1)  # (N_val, 1400)
    feat_test  = torch.cat(all_feat_test,  dim=1)  # (N_test, 1400)
    tgt_train  = torch.cat(tgt_train_list, dim=0)  # (N_train, 7)
    tgt_val    = torch.cat(tgt_val_list,   dim=0)  # (N_val, 7)
    tgt_test   = torch.cat(tgt_test_list,  dim=0)  # (N_test, 7)
    del all_feat_train, all_feat_val, all_feat_test
    del eeg_train_scans, eeg_val_scans, eeg_test_scans
    gc.collect()

    print(f'  feat_train={feat_train.shape}, feat_val={feat_val.shape}, feat_test={feat_test.shape}')
    print(f'  tgt_train={tgt_train.shape}')

    # ── Train Joint Gaussian Head ────────────────────────────────────────────
    print('\n[2] Training Joint Multivariate Gaussian Head (300 epochs)...')
    gauss_model, mu_g, sig_g = train_gaussian_head(
        feat_train, tgt_train, feat_val, tgt_val, epochs=300
    )

    print('\n[3] Evaluating Gaussian Head (N=50 samples):')
    gauss_results = eval_gaussian_head(gauss_model, mu_g, sig_g, feat_test, tgt_test, n_samples=50)
    gauss_avg_r = float(np.mean([gauss_results[d]['R'] for d in ROI_DISPLAY]))
    gauss_avg_crps = float(np.mean([gauss_results[d]['CRPS'] for d in ROI_DISPLAY]))
    gauss_avg_cover = float(np.mean([gauss_results[d]['Coverage_90'] for d in ROI_DISPLAY]))
    gauss_avg_nll   = float(np.mean([gauss_results[d]['NLL']  for d in ROI_DISPLAY]))
    gauss_cov_mae   = gauss_results.get('cov_mae', float('nan'))
    print(f'  Gaussian Head Avg.R={gauss_avg_r:.3f}  Avg.CRPS={gauss_avg_crps:.4f}  '
          f'Avg.Cover90={gauss_avg_cover:.3f}  Avg.NLL={gauss_avg_nll:.4f}  '
          f'CovMAE={gauss_cov_mae:.4f}')

    # ── FM v3b Temperature Calibration ──────────────────────────────────────
    print('\n[4] FM v3b temperature scaling calibration...')
    if not os.path.exists(V3B_CKPT):
        print('  FM v3b checkpoint not found, skipping calibration.')
        best_T = 1.0
        fm_results_cal = None
    else:
        ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
        mu_fm  = np.array(ckpt['mu'])
        sig_fm = np.array(ckpt['sig'])
        head = MultiSourceFMHead(feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7)
        head.load_state_dict(ckpt['head'])
        head.eval().to(DEVICE)

        best_T = find_fm_temperature(head, mu_fm, sig_fm, feat_val, tgt_val)

        print(f'\n[5] FM v3b with T={best_T:.1f} (N=50 trajectories):')
        fm_samples, fm_cover_cal = eval_fm_with_temperature(
            head, mu_fm, sig_fm, feat_test, tgt_test, temperature=best_T, n_samples=50)

        true = tgt_test.numpy()
        fm_results_cal = {}
        for j, disp in enumerate(ROI_DISPLAY):
            roi_s  = fm_samples[:, :, j]
            roi_tr = (true[:, j] - mu_fm[j]) / sig_fm[j]
            crps  = crps_ensemble(roi_s, roi_tr)
            cover = interval_coverage(roi_s, roi_tr, alpha=0.1)
            nll   = nll_gaussian_samples(roi_s, roi_tr)
            pred  = roi_s.mean(0) * sig_fm[j] + mu_fm[j]
            r_val, _ = pearsonr(pred, true[:, j])
            fm_results_cal[disp] = {'R': float(r_val), 'CRPS': float(crps),
                                     'Coverage_90': float(cover), 'NLL': float(nll)}
            print(f'  {disp:<20}: R={r_val:.3f}  CRPS={crps:.4f}  '
                  f'Cover90={cover:.3f}  NLL={nll:.4f}')

        # Covariance recovery (FM calibrated)
        pred_mean = fm_samples.mean(0) * sig_fm[None, :] + mu_fm[None, :]
        pred_corr = np.corrcoef(pred_mean.T)
        true_corr = np.corrcoef(true.T)
        off_diag  = ~np.eye(7, dtype=bool)
        fm_cov_mae_cal = float(np.mean(np.abs(pred_corr[off_diag] - true_corr[off_diag])))
        fm_avg_r_cal   = float(np.mean([fm_results_cal[d]['R'] for d in ROI_DISPLAY]))
        fm_avg_crps_cal= float(np.mean([fm_results_cal[d]['CRPS'] for d in ROI_DISPLAY]))
        fm_avg_cover_cal = float(np.mean([fm_results_cal[d]['Coverage_90'] for d in ROI_DISPLAY]))
        fm_avg_nll_cal   = float(np.mean([fm_results_cal[d]['NLL'] for d in ROI_DISPLAY]))
        print(f'  FM (T={best_T:.1f}) Avg.R={fm_avg_r_cal:.3f}  Avg.CRPS={fm_avg_crps_cal:.4f}  '
              f'Avg.Cover90={fm_avg_cover_cal:.3f}  Avg.NLL={fm_avg_nll_cal:.4f}  '
              f'CovMAE={fm_cov_mae_cal:.4f}')
        fm_results_cal['cov_mae'] = fm_cov_mae_cal
        fm_results_cal['temperature'] = best_T

    # ── Summary Table ─────────────────────────────────────────────────────────
    print('\n' + '='*90)
    print('  Probabilistic Comparison: FM v3b (calibrated) vs Joint Gaussian Head')
    print('='*90)
    print(f"{'Method':<30} {'Avg.R':>8} {'Avg.CRPS':>10} {'Cover90':>10} {'Avg.NLL':>10} {'CovMAE':>10}")
    print('-'*90)

    # FM v3b original (T=1.0)
    print(f"{'FM v3b (T=1.0, original)':<30} {'0.361':>8} {'0.755':>10} {'0.280':>10} {'14.87':>10} {'0.0585':>10}")
    # FM calibrated
    if fm_results_cal:
        print(f"{'FM v3b (T='+f'{best_T:.1f}'+', calibrated)':<30} "
              f"{fm_avg_r_cal:>8.3f} {fm_avg_crps_cal:>10.4f} "
              f"{fm_avg_cover_cal:>10.3f} {fm_avg_nll_cal:>10.4f} {fm_cov_mae_cal:>10.4f}")
    # Gaussian
    print(f"{'Joint Gaussian Head':<30} {gauss_avg_r:>8.3f} {gauss_avg_crps:>10.4f} "
          f"{gauss_avg_cover:>10.3f} {gauss_avg_nll:>10.4f} {gauss_cov_mae:>10.4f}")
    print('='*90)

    # ── Save Results ──────────────────────────────────────────────────────────
    output = {
        'gaussian_head': gauss_results,
        'fm_v3b_calibrated': fm_results_cal,
        'summary': {
            'gaussian': {'avg_r': gauss_avg_r, 'avg_crps': gauss_avg_crps,
                         'avg_cover90': gauss_avg_cover, 'avg_nll': gauss_avg_nll,
                         'cov_mae': gauss_cov_mae},
            'fm_calibrated': {
                'avg_r': fm_avg_r_cal if fm_results_cal else None,
                'avg_crps': fm_avg_crps_cal if fm_results_cal else None,
                'avg_cover90': fm_avg_cover_cal if fm_results_cal else None,
                'avg_nll': fm_avg_nll_cal if fm_results_cal else None,
                'cov_mae': fm_cov_mae_cal if fm_results_cal else None,
                'temperature': best_T,
            }
        }
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
