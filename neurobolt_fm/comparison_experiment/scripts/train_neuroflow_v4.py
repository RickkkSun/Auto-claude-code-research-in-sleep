"""
train_neuroflow_v4.py — Residual Flow Matching for improved CRPS.

Key improvement over v3b:
  1. Linear mean predictor: μ_hat = Linear(1400D features)
  2. Residual FM: models P(fMRI - μ_hat | features) instead of P(fMRI | features)
  3. Post-hoc tau optimization: shrink sample spread to minimize validation CRPS
  4. Model selection by validation CRPS (not MSE)

Expected gain: residuals have ~80-85% of original variance → CRPS reduction ~10-15%
Combined with tau post-hoc: CRPS should reach ~0.24-0.25 from baseline 0.304.

Architecture:
  EEG → 7×NeuroBOLT (frozen) → 1400D features
  → LinearMean: 1400 → 7      (predict conditional mean)
  → ResidualFM: model P(fMRI - LinearMean | 1400D)  [smaller spread]
  → Inference: LinearMean(features) + ResidualFM_sample * tau
"""

import sys, os, gc, json, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr
from einops import rearrange

BASE     = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400

ROI_DISPLAY = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']

print(f'Device: {DEVICE}')


# ─────────────────────────────────────────────────────────────────────────────
# Load NeuroBOLT feature cache
# ─────────────────────────────────────────────────────────────────────────────

def load_cache():
    cache_file = os.path.join(CACHE_DIR, 'features_conf.pt')
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f'Feature cache not found: {cache_file}')
    print('Loading feature cache...', flush=True)
    d = torch.load(cache_file, map_location='cpu', weights_only=False)
    return (d['feat_tr'].float(), d['tgt_tr'].float(),
            d['feat_cal'].float(), d['tgt_cal'].float(),
            d['feat_te'].float(), d['tgt_te'].float())


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half-1, 1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualFMHead(nn.Module):
    """
    FM head for residuals. Takes (features, x_t - mu_hat) and predicts velocity.
    Identical architecture to v3b MultiSourceFMHead, but trained on residuals.
    """
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512,
                 n_rois=7, time_dim=32):
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
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_rois),
        )

    def forward(self, features, x_t, t):
        f = self.feat_proj(features)
        e = self.time_emb(t)
        xi = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f, xi, e], dim=-1))

    @torch.no_grad()
    def sample(self, features, n_samples=50, num_steps=100):
        """Return [n_samples, B, n_rois] residual samples."""
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


# ─────────────────────────────────────────────────────────────────────────────
# OT-CFM loss
# ─────────────────────────────────────────────────────────────────────────────

def ot_cfm_loss(head, features, targets, sigma_min=0.001):
    """Optimal Transport CFM loss on residuals."""
    B     = targets.shape[0]
    x0    = torch.randn_like(targets)               # source noise
    x1    = targets                                 # target (residuals, zero-mean)
    t     = torch.rand(B, 1, device=targets.device)

    # Straight-line interpolation (OT path): x_t = (1-t)*x0 + t*x1
    x_t   = (1 - t) * x0 + t * x1
    # Small Gaussian noise for stability
    x_t   = x_t + sigma_min * torch.randn_like(x_t)
    # Target velocity for OT path: v* = x1 - x0
    v_star = x1 - x0

    v_pred = head(features, x_t, t)
    return F.mse_loss(v_pred, v_star)


# ─────────────────────────────────────────────────────────────────────────────
# CRPS computation
# ─────────────────────────────────────────────────────────────────────────────

def crps_score(samples, targets, exact=True):
    """
    CRPS via energy score: E|Y-y| - 0.5*E|Y-Y'|
    samples: [K, N, D], targets: [N, D]
    exact=True: all-pairs (no randomness, unbiased at final eval)
    exact=False: random-pair estimate (fast, used during training)
    """
    K, N, D = samples.shape
    e1 = (samples - targets.unsqueeze(0)).abs().mean()
    if exact:
        diff = (samples.unsqueeze(1) - samples.unsqueeze(0)).abs()  # [K, K, N, D]
        e2 = 0.5 * diff.mean()
    else:
        perm = torch.randperm(K)
        e2 = 0.5 * (samples - samples[perm]).abs().mean()
    return (e1 - e2).item()


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc variance shrinkage (tau optimization)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_tau(samples_raw, targets, n_tau=40):
    """
    Find optimal scale tau that minimizes CRPS on validation set.
    samples_raw: [K, N, D] — FM samples (residuals, not yet tau-scaled)
    targets:     [N, D] — validation targets (residuals)
    Returns: best tau (scalar)
    """
    mu    = samples_raw.mean(0, keepdim=True)   # [1, N, D] sample mean
    taus  = torch.linspace(0.3, 1.5, n_tau)
    best_crps, best_tau = 1e9, 1.0

    for tau in taus:
        scaled = mu + tau * (samples_raw - mu)   # [K, N, D]
        c = crps_score(scaled, targets)
        if c < best_crps:
            best_crps = c
            best_tau  = float(tau)

    print(f'  tau optimization: best tau={best_tau:.3f}  CRPS={best_crps:.4f}')
    return best_tau


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_residual_fm(feat_tr, tgt_tr, feat_cal, tgt_cal,
                      epochs=400, lr=3e-4, bs=128):
    N = len(feat_tr)
    print(f'\n[1] Training Linear Mean Predictor...', flush=True)

    # --- Linear mean predictor ---
    lin = nn.Linear(FEAT_DIM, N_ROIS).to(DEVICE)
    opt_lin = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(100):
        perm = torch.randperm(N)
        for i in range(0, N, 256):
            idx = perm[i:i+256]
            loss = F.mse_loss(lin(feat_tr[idx].to(DEVICE)), tgt_tr[idx].to(DEVICE))
            opt_lin.zero_grad(); loss.backward(); opt_lin.step()
    lin.eval()

    # Compute residuals
    with torch.no_grad():
        mu_tr  = torch.cat([lin(feat_tr[i:i+512].to(DEVICE)).cpu() for i in range(0, N, 512)])
        mu_cal = torch.cat([lin(feat_cal[i:i+512].to(DEVICE)).cpu() for i in range(0, len(feat_cal), 512)])
    res_tr  = tgt_tr  - mu_tr.detach()
    res_cal = tgt_cal - mu_cal.detach()

    # Check linear R
    for j in range(N_ROIS):
        r, _ = pearsonr(mu_tr[:, j].numpy(), tgt_tr[:, j].numpy())
        print(f'  Linear R [{ROI_DISPLAY[j]}] = {r:.4f}')

    print(f'\n  Residual variance explained: {1 - res_tr.var(0).mean() / tgt_tr.var(0).mean():.3f}')

    # Normalize residuals
    sig = res_tr.std(0).clamp(min=1e-8)
    res_tr_n  = res_tr  / sig
    res_cal_n = res_cal / sig

    print(f'\n[2] Training Residual FM (residuals, zero-mean)...', flush=True)

    head = ResidualFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, math.ceil(N / bs))
    )

    best_crps, best_state, no_imp = 1e9, None, 0
    patience = 40

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            f  = feat_tr[idx].to(DEVICE)
            y  = res_tr_n[idx].to(DEVICE)
            loss = ot_cfm_loss(head, f, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        # Validation CRPS every 20 epochs
        if (ep + 1) % 20 == 0:
            head.eval()
            with torch.no_grad():
                samps = head.sample(feat_cal.to(DEVICE), n_samples=20, num_steps=50)
                # Un-normalize residuals
                samps = samps * sig.to(DEVICE)
                res_cal_device = res_cal.to(DEVICE)
                val_crps = crps_score(samps, res_cal_device, exact=False)  # fast during training
            print(f'  ep {ep+1}/{epochs}  val_residual_CRPS={val_crps:.4f}  best={best_crps:.4f}')
            if val_crps < best_crps:
                best_crps = val_crps
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                print(f'  Early stop at ep {ep+1}')
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    print(f'\n  Best val residual CRPS = {best_crps:.4f}')

    return lin, head, sig


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(lin, head, sig, feat, tgt, tau=1.0, n_samples=50, num_steps=100):
    """Full evaluation: R, CRPS, FC-MAE. Returns samples for downstream CI testing."""
    # Sample residuals
    samps_res = head.sample(feat.to(DEVICE), n_samples=n_samples,
                            num_steps=num_steps)   # [K, N, 7] (normalized residuals)
    samps_res = samps_res * sig.to(DEVICE)         # un-normalize

    # Conditional mean
    mu_hat = torch.cat([lin(feat[i:i+512].to(DEVICE)).cpu() for i in range(0, len(feat), 512)])

    # Apply tau scaling
    mu_samps = samps_res.mean(0)   # [N, 7]
    samps_res_scaled = mu_samps.to(DEVICE).unsqueeze(0) + tau * (samps_res - mu_samps.to(DEVICE).unsqueeze(0))

    # Total samples: mu_hat + scaled_residuals
    samps_total = mu_hat.to(DEVICE).unsqueeze(0) + samps_res_scaled   # [K, N, 7]
    samps_cpu   = samps_total.cpu()
    pred_mean   = samps_cpu.mean(0)

    # Pearson R
    roi_r = []
    for j in range(N_ROIS):
        r, _ = pearsonr(pred_mean[:, j].numpy(), tgt[:, j].numpy())
        roi_r.append(float(r))

    # CRPS — exact all-pairs at test time (no random approximation)
    crps = crps_score(samps_cpu, tgt, exact=True)

    # FC-MAE (functional connectivity — sample-mean correlation matrix)
    def corr_mat(x):
        x = x - x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True).clamp(min=1e-8)
        xn = x / std
        return (xn.T @ xn) / (len(xn) - 1)
    fc_pred = corr_mat(pred_mean)
    fc_tgt  = corr_mat(tgt)
    fc_mae  = (fc_pred - fc_tgt).abs().mean().item()

    return {
        'avg_r'   : float(np.mean(roi_r)),
        'roi_r'   : roi_r,
        'crps'    : crps,
        'fc_mae'  : fc_mae,
        '_samples': samps_cpu,   # kept for bootstrap CI; not serialized to JSON
    }


def bootstrap_ci_nf(samples, targets, n_boot=500, alpha=0.05, seed=42):
    """
    Per-example (iid) bootstrap 95% CI for CRPS and avg_R.
    For paired significance tests, use paired_bootstrap_test() instead.
    samples: [K, N, 7], targets: [N, 7]
    """
    rng = np.random.default_rng(seed)
    N   = targets.shape[0]
    crps_vals, r_vals = [], []
    pred_mean_full = samples.float().mean(0)   # [N, 7]

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        s_b = samples[:, idx, :]
        t_b = targets[idx]
        pm  = pred_mean_full[idx]
        crps_vals.append(crps_score(s_b, t_b, exact=True))
        r_vals.append(float(np.mean([
            pearsonr(pm[:, j].numpy(), t_b[:, j].numpy())[0]
            for j in range(N_ROIS)
        ])))

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        'crps_ci': (float(np.quantile(crps_vals, lo)), float(np.quantile(crps_vals, hi))),
        'r_ci'   : (float(np.quantile(r_vals,    lo)), float(np.quantile(r_vals,    hi))),
    }


def _per_example_crps_nf(samps, tgt):
    """Per-example CRPS [N] — for paired testing."""
    e1 = (samps - tgt.unsqueeze(0)).abs().mean(dim=(0, 2))   # [N]
    diff = (samps.unsqueeze(1) - samps.unsqueeze(0)).abs().mean(dim=(0, 1, 3))  # [N]
    return e1 - 0.5 * diff


def paired_bootstrap_test(samples_a, samples_b, targets, n_boot=1000, alpha=0.05, seed=0):
    """
    Paired bootstrap significance test (centered bootstrap): H0: E[CRPS_A] = E[CRPS_B].
    One-sided: A is significantly better if p_value < alpha (CRPS_A < CRPS_B).

    Steps:
      d_i = CRPS_A_i - CRPS_B_i
      obs_delta = mean(d)                          (negative if A is better)
      d_0 = d - obs_delta                          (center at 0 under H0)
      t_boot = mean(d_0[resample])                 (bootstrap null distribution)
      p_value = fraction of t_boot <= obs_delta    (one-sided)
    """
    rng = np.random.default_rng(seed)
    N   = targets.shape[0]

    score_a = _per_example_crps_nf(samples_a, targets).numpy()
    score_b = _per_example_crps_nf(samples_b, targets).numpy()
    d = score_a - score_b

    obs_delta = float(d.mean())
    d_0 = d - obs_delta   # center at 0 under H0

    boot_means = np.array([d_0[rng.integers(0, N, size=N)].mean() for _ in range(n_boot)])
    p_val = float((boot_means <= obs_delta).mean())
    return {'obs_delta': obs_delta, 'p_value': p_val, 'significant': p_val < alpha}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',    type=str, default='pooled', choices=['intra','pooled'],
                        help='Evaluation mode (affects result key naming for table assembly)')
    parser.add_argument('--epochs',  type=int, default=400)
    parser.add_argument('--lr',      type=float, default=3e-4)
    parser.add_argument('--n_boot',  type=int, default=500, help='Bootstrap iterations for CI')
    args = parser.parse_args()

    print(f'NeuroFlow v4 — mode={args.mode}  epochs={args.epochs}', flush=True)
    feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te = load_cache()
    print(f'  train={len(feat_tr)}, cal={len(feat_cal)}, test={len(feat_te)}')

    # Train residual FM
    lin, head, sig = train_residual_fm(feat_tr, tgt_tr, feat_cal, tgt_cal,
                                       epochs=args.epochs, lr=args.lr, bs=128)

    # Post-hoc tau optimization on calibration set
    print('\n[3] Tau optimization...', flush=True)
    with torch.no_grad():
        cal_res_samps = head.sample(feat_cal.to(DEVICE), n_samples=50, num_steps=100)
        cal_res_samps = cal_res_samps * sig.to(DEVICE)
    mu_cal = torch.cat([lin(feat_cal[i:i+512].to(DEVICE)).cpu() for i in range(0, len(feat_cal), 512)])
    cal_res_cpu = cal_res_samps.cpu()
    cal_tgt_res = tgt_cal - mu_cal

    tau = optimize_tau(cal_res_cpu, cal_tgt_res)

    # Test evaluation
    print('\n[4] Test evaluation (exact CRPS + bootstrap CI)...', flush=True)
    results_tau = evaluate(lin, head, sig, feat_te, tgt_te, tau=tau, n_samples=50)
    samps_te = results_tau.pop('_samples')  # remove non-serializable tensor

    # Bootstrap CI
    print(f'\n[5] Bootstrap CI ({args.n_boot} iterations)...', flush=True)
    ci = bootstrap_ci_nf(samps_te, tgt_te, n_boot=args.n_boot)
    results_tau['crps_ci'] = ci['crps_ci']
    results_tau['r_ci']    = ci['r_ci']
    results_tau['tau']     = tau
    results_tau['mode']    = args.mode

    print('\n=== RESULTS ===')
    print(f'With tau={tau:.3f}: R={results_tau["avg_r"]:.4f}  '
          f'CRPS={results_tau["crps"]:.4f} [{ci["crps_ci"][0]:.4f},{ci["crps_ci"][1]:.4f}]  '
          f'FC-MAE={results_tau["fc_mae"]:.4f}')
    print(f'  R 95% CI: [{ci["r_ci"][0]:.4f}, {ci["r_ci"][1]:.4f}]')

    print('\nROI breakdown (tau-scaled):')
    for name, r in zip(ROI_DISPLAY, results_tau['roi_r']):
        print(f'  {name:<14}: R={r:.4f}')

    # Save results — key format: neuroflow_v4_{mode}_tau
    # build_paper_tables.py looks for neuroflow_{mode} under system results
    result_key = f'neuroflow_v4_{args.mode}_tau'
    out_path = os.path.join(OUT_DIR, 'neuroflow_v4_results.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing[result_key] = results_tau
    # Also write the key build_paper_tables.py expects directly
    existing[f'neuroflow_{args.mode}'] = {
        'avg_r'    : results_tau['avg_r'],
        'roi_r'    : results_tau['roi_r'],
        'roi_r_mean': results_tau['roi_r'],
        'crps'     : results_tau['crps'],
        'fc_mae'   : results_tau['fc_mae'],
        'crps_ci'  : results_tau['crps_ci'],
        'r_ci'     : results_tau['r_ci'],
        'mode'     : args.mode,
    }
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults saved to {out_path}  (key: {result_key})')
