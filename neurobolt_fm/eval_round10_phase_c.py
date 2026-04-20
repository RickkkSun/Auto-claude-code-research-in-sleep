#!/usr/bin/env python3
"""
eval_round10_phase_c.py
Phase C fixes for Round 1 of new loop:
1. MDN baseline (Mixture Density Network, K=3 Gaussians)
2. FM sample sensitivity (N=10, 20, 50, 100 samples)
3. Joint coverage evaluation (all 7 ROIs simultaneously covered)
4. Calibration reliability diagram data
5. Updated comprehensive comparison JSON

GPU required for FM parts.
"""

import sys, os, json, gc, warnings, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
V3B_CKPT  = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
OUT_JSON  = f'{BASE}/phase_c_results.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400

print(f'Device: {DEVICE}')
print(f'Checkpoint: {V3B_CKPT}')


# ─────────────────────────────────────────────────────────────────────────────
# Load feature cache
# ─────────────────────────────────────────────────────────────────────────────
def load_cache():
    cache_file = os.path.join(CACHE_DIR, 'features_conf.pt')
    d = torch.load(cache_file, map_location='cpu', weights_only=False)
    feat_tr  = d['feat_tr'].float()
    tgt_tr   = d['tgt_tr'].float()
    feat_cal = d['feat_cal'].float()
    tgt_cal  = d['tgt_cal'].float()
    feat_te  = d['feat_te'].float()
    tgt_te   = d['tgt_te'].float()
    print(f'Cache: tr={len(feat_tr)}, cal={len(feat_cal)}, te={len(feat_te)}')
    # Combine train+cal (90%), split test 50/50
    feat_all_tr = torch.cat([feat_tr, feat_cal], 0)
    tgt_all_tr  = torch.cat([tgt_tr,  tgt_cal],  0)
    n_te = len(feat_te)
    nh   = n_te // 2
    return (feat_all_tr, tgt_all_tr,
            feat_te[:nh], tgt_te[:nh],   # heldout-cal
            feat_te[nh:], tgt_te[nh:])   # final-test


# ─────────────────────────────────────────────────────────────────────────────
# FM v3b model (same architecture as eval_conformalized_joint.py)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer('freq', freq)

    def forward(self, t):
        if t.dim() == 0: t = t.unsqueeze(0)
        x = t[:, None] * self.freq[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class FMHead(nn.Module):
    def __init__(self, n_roi=7, feat_dim=1400, hidden=512, time_dim=64,
                 n_layers=4, dropout=0.0):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(time_dim)
        self.in_proj = nn.Linear(feat_dim + n_roi + time_dim, hidden)
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
        self.mid = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, n_roi)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x_feat, x_t, t):
        te = self.time_emb(t)
        if te.shape[0] == 1 and x_feat.shape[0] > 1:
            te = te.expand(x_feat.shape[0], -1)
        h = torch.cat([x_feat, x_t, te], dim=-1)
        h = F.silu(self.in_proj(h))
        h = self.mid(h)
        return self.out(h)


def load_fm_v3b():
    fm = FMHead(n_roi=N_ROIS, feat_dim=FEAT_DIM, hidden=512, time_dim=64,
                n_layers=4, dropout=0.0).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    state = ckpt[state_key] if state_key else ckpt
    fm.load_state_dict(state, strict=True)
    fm.eval()
    for p in fm.parameters(): p.requires_grad_(False)
    print('  FM v3b loaded.')
    return fm


@torch.no_grad()
def fm_sample(fm, feat, fm_mu, fm_sig, n_samples=50, n_steps=20, bs=128):
    """ODE integration (Euler, n_steps) to generate n_samples."""
    N = len(feat)
    all_samples = []  # [n_samples, N, 7]
    for _ in range(n_samples):
        preds = []
        for i in range(0, N, bs):
            xf = feat[i:i+bs].to(DEVICE)
            B  = xf.size(0)
            x  = torch.randn(B, N_ROIS, device=DEVICE)
            dt = 1.0 / n_steps
            for step in range(n_steps):
                t_val = step * dt * torch.ones(B, device=DEVICE)
                v = fm(xf, x, t_val)
                x = x + v * dt
            x_orig = x * fm_sig.to(DEVICE) + fm_mu.to(DEVICE)
            preds.append(x_orig.cpu())
        all_samples.append(torch.cat(preds, 0))
    return torch.stack(all_samples, 0)  # [n_samples, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# MDN (Mixture Density Network) — K Gaussian components
# ─────────────────────────────────────────────────────────────────────────────
class MDNHead(nn.Module):
    """Multivariate diagonal-Gaussian MDN with K components."""
    def __init__(self, in_dim=1400, hidden=512, n_roi=7, K=3, dropout=0.15):
        super().__init__()
        self.K = K
        self.n_roi = n_roi
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2), nn.GELU(), nn.Dropout(dropout),
        )
        self.pi_head    = nn.Linear(hidden//2, K)              # mixing weights
        self.mu_head    = nn.Linear(hidden//2, K * n_roi)      # component means
        self.sigma_head = nn.Linear(hidden//2, K * n_roi)      # log-std

    def forward(self, x):
        h = self.base(x)
        pi = F.softmax(self.pi_head(h), dim=-1)                     # [B, K]
        mu = self.mu_head(h).view(-1, self.K, self.n_roi)            # [B, K, 7]
        log_sigma = self.sigma_head(h).view(-1, self.K, self.n_roi)  # [B, K, 7]
        sigma = F.softplus(log_sigma) + 1e-5                         # [B, K, 7]
        return pi, mu, sigma

    def nll_loss(self, x, y):
        """Negative log-likelihood under MDN."""
        pi, mu, sigma = self.forward(x)                              # [B,K], [B,K,7], [B,K,7]
        y_exp = y.unsqueeze(1).expand_as(mu)                         # [B, K, 7]
        # Log prob under each component (diagonal Gaussian)
        log_p = -0.5 * ((y_exp - mu) / sigma).pow(2) - sigma.log()  # [B, K, 7]
        log_p = log_p.sum(-1)                                        # [B, K]
        # Mix: log sum_k pi_k * N(y | mu_k, sigma_k)
        log_mix = torch.logsumexp(log_p + pi.log(), dim=-1)          # [B]
        return -log_mix.mean()

    def sample(self, x, n_samples=50):
        """Draw samples from MDN: [n_samples, B, 7]."""
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)                           # [B,K], [B,K,7], [B,K,7]
            B = x.size(0)
            all_samp = []
            for _ in range(n_samples):
                # sample component
                k = torch.multinomial(pi, 1).squeeze(-1)             # [B]
                mu_k = mu[torch.arange(B), k]                        # [B, 7]
                sig_k = sigma[torch.arange(B), k]                    # [B, 7]
                z = torch.randn_like(mu_k)
                all_samp.append(mu_k + sig_k * z)
            return torch.stack(all_samp, 0)                           # [n_samples, B, 7]

    def mean(self, x):
        """Predictive mean (mixture mean)."""
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)                           # [B,K], [B,K,7]
            return (pi.unsqueeze(-1) * mu).sum(1)                     # [B, 7]


def train_mdn(feat_tr, tgt_tr, K=3, epochs=120, lr=3e-4, bs=128):
    model = MDNHead(in_dim=FEAT_DIM, hidden=512, n_roi=N_ROIS, K=K).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(feat_tr)
    for ep in range(epochs):
        idx = torch.randperm(N)
        total_loss = 0
        model.train()
        for i in range(0, N, bs):
            b = idx[i:i+bs]
            x = feat_tr[b].to(DEVICE)
            y = tgt_tr[b].to(DEVICE)
            loss = model.nll_loss(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
        sched.step()
        if (ep+1) % 30 == 0:
            print(f'    MDN Epoch {ep+1}/{epochs} NLL={total_loss/N:.4f}')
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Conformal calibration utilities
# ─────────────────────────────────────────────────────────────────────────────
def conformalize_samples(samples_cal, tgt_cal, samples_te, tgt_te, alpha=0.10):
    """
    Marginal split conformal calibration from samples.
    samples_cal: [n_samp, N_cal, 7]
    Returns: per_roi_q, joint_coverage, marginal_coverage
    """
    N_cal = tgt_cal.shape[0]
    N_te  = tgt_te.shape[0]

    # Compute residuals on cal set: |y_i - median(samples_i)|
    med_cal = samples_cal.median(0).values  # [N_cal, 7]
    resid_cal = (tgt_cal - med_cal).abs()   # [N_cal, 7]

    # Conformal quantile per ROI
    level = np.ceil((N_cal + 1) * (1 - alpha)) / N_cal
    level = min(level, 1.0)
    q_per_roi = torch.quantile(resid_cal, level, dim=0)  # [7]

    # Evaluate on test set
    med_te = samples_te.median(0).values   # [N_te, 7]
    lo = med_te - q_per_roi
    hi = med_te + q_per_roi
    covered = (tgt_te >= lo) & (tgt_te <= hi)   # [N_te, 7]

    marginal_cov = covered.float().mean(0).numpy()  # [7]
    joint_cov    = covered.all(-1).float().mean().item()
    avg_marginal = float(marginal_cov.mean())
    width = float((hi - lo).mean().item())

    return {
        'q_per_roi'      : q_per_roi.tolist(),
        'marginal_cov'   : marginal_cov.tolist(),
        'avg_marginal_cov': avg_marginal,
        'joint_cov'      : joint_cov,
        'width'          : width,
    }


def compute_r_from_samples(samples_te, tgt_te):
    """Compute per-ROI R using predictive mean (median of samples)."""
    pred = samples_te.median(0).values.numpy()  # [N, 7]
    y    = tgt_te.numpy()
    r_vals = []
    for ri in range(N_ROIS):
        r, _ = pearsonr(pred[:, ri], y[:, ri])
        r_vals.append(float(r))
    return r_vals, float(np.mean(r_vals))


def compute_crps_from_samples(samples_te, tgt_te, n_mc=1000):
    """Energy score CRPS approximation from samples."""
    n_samp = samples_te.shape[0]
    N = tgt_te.shape[0]
    # Use subset for speed
    idx = torch.randperm(n_samp)[:min(n_samp, 50)]
    samp = samples_te[idx]  # [M, N, 7]
    y = tgt_te               # [N, 7]
    # CRPS per ROI = E|X - y| - 0.5 * E|X - X'|
    crps_vals = []
    for ri in range(N_ROIS):
        s = samp[:, :, ri]     # [M, N]
        y_r = y[:, ri]         # [N]
        e_xy = (s - y_r.unsqueeze(0)).abs().mean(0).mean().item()
        # Pairwise
        idx2 = torch.randperm(len(idx))[:len(idx)//2]
        idx3 = torch.randperm(len(idx))[:len(idx)//2]
        e_xx = (samp[idx2, :, ri] - samp[idx3, :, ri]).abs().mean(0).mean().item()
        crps_vals.append(e_xy - 0.5 * e_xx)
    return crps_vals, float(np.mean(crps_vals))


def compute_cov_mae_from_samples(samples_te, tgt_te):
    """CovMAE: |pred_covariance - empirical_covariance| (Frobenius mean)."""
    n_samp = samples_te.shape[0]
    # Predicted covariance: across samples at each time point, then avg over time
    # Shape: [n_samp, N, 7] → for each time point, cov over samples
    # Simpler: use sample covariance across the N test points
    pred_mean = samples_te.mean(0)   # [N, 7]
    residuals = pred_mean - tgt_te   # [N, 7]
    pred_cov  = torch.cov(pred_mean.T)   # [7, 7]
    true_cov  = torch.cov(tgt_te.T)     # [7, 7]
    return float((pred_cov - true_cov).abs().mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# FM normalization stats (from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────
def get_fm_norm_stats(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    fm_mu  = ckpt.get('fm_mu',  None)
    fm_sig = ckpt.get('fm_sig', None)
    if fm_mu is None:
        # Try to recover from training data
        return None, None
    return fm_mu, fm_sig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    feat_all_tr, tgt_all_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte = load_cache()
    print(f'Splits: train={len(feat_all_tr)}, heldout-cal={len(feat_hcal)}, final-test={len(feat_fte)}')

    results = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1. MDN baseline (K=3)
    # ────────────────────────────────────────────────────────────────────────
    print('\n[1/3] Training MDN (K=3)...')
    mdn = train_mdn(feat_all_tr, tgt_all_tr, K=3, epochs=120, lr=3e-4)
    mdn.eval()
    with torch.no_grad():
        # Get samples for conformal calibration
        bs = 256
        cal_samps = []
        for i in range(0, len(feat_hcal), bs):
            s = mdn.sample(feat_hcal[i:i+bs].to(DEVICE), n_samples=50).cpu()
            cal_samps.append(s)
        cal_samps = torch.cat(cal_samps, 1)  # [50, N_cal, 7]

        te_samps = []
        for i in range(0, len(feat_fte), bs):
            s = mdn.sample(feat_fte[i:i+bs].to(DEVICE), n_samples=50).cpu()
            te_samps.append(s)
        te_samps = torch.cat(te_samps, 1)    # [50, N_te, 7]

    conf_mdn = conformalize_samples(cal_samps, tgt_hcal, te_samps, tgt_fte, alpha=0.10)
    r_mdn, avg_r_mdn = compute_r_from_samples(te_samps, tgt_fte)
    crps_mdn, avg_crps_mdn = compute_crps_from_samples(te_samps, tgt_fte)
    cov_mae_mdn = compute_cov_mae_from_samples(te_samps, tgt_fte)

    results['mdn_k3'] = {
        'avg_r': avg_r_mdn,
        'per_roi_r': r_mdn,
        'avg_crps': avg_crps_mdn,
        'per_roi_crps': crps_mdn,
        'cov_mae': cov_mae_mdn,
        **conf_mdn,
    }
    print(f'  MDN Avg.R={avg_r_mdn:.3f}, CRPS={avg_crps_mdn:.3f}, '
          f'CovMAE={cov_mae_mdn:.3f}, Cover90={conf_mdn["avg_marginal_cov"]:.3f}')
    print(f'  Joint coverage={conf_mdn["joint_cov"]:.3f}')

    # ────────────────────────────────────────────────────────────────────────
    # 2. FM v3b: sample sensitivity (N=10, 20, 50, 100)
    # ────────────────────────────────────────────────────────────────────────
    print('\n[2/3] FM v3b sample sensitivity...')
    if not os.path.exists(V3B_CKPT):
        print('  FM checkpoint not found, skipping.')
        results['fm_sensitivity'] = None
    else:
        fm = load_fm_v3b()

        # Get normalization stats from cache or estimate from data
        # Use tgt_all_tr to estimate mu, sig
        fm_mu  = tgt_all_tr.mean(0)   # [7]
        fm_sig = tgt_all_tr.std(0)    # [7]
        print(f'  FM norm: mu={fm_mu.tolist()}, sig={fm_sig.tolist()}')

        # Generate max samples (100) once, then subsample
        print('  Generating 100 FM samples on cal set...')
        fm_cal_samps  = fm_sample(fm, feat_hcal, fm_mu, fm_sig, n_samples=100, n_steps=20)
        print('  Generating 100 FM samples on test set...')
        fm_te_samps   = fm_sample(fm, feat_fte,  fm_mu, fm_sig, n_samples=100, n_steps=20)

        sensitivity = {}
        for n_samp in [10, 20, 50, 100]:
            print(f'  N={n_samp}...')
            samp_cal = fm_cal_samps[:n_samp]
            samp_te  = fm_te_samps[:n_samp]
            conf = conformalize_samples(samp_cal, tgt_hcal, samp_te, tgt_fte, alpha=0.10)
            r_vals, avg_r = compute_r_from_samples(samp_te, tgt_fte)
            crps_vals, avg_crps = compute_crps_from_samples(samp_te, tgt_fte)
            cov_mae = compute_cov_mae_from_samples(samp_te, tgt_fte)
            sensitivity[str(n_samp)] = {
                'avg_r': avg_r, 'avg_crps': avg_crps,
                'cov_mae': cov_mae, **conf,
            }
            print(f'    N={n_samp}: R={avg_r:.3f}, CRPS={avg_crps:.3f}, '
                  f'CovMAE={cov_mae:.3f}, MargCov={conf["avg_marginal_cov"]:.3f}, '
                  f'JointCov={conf["joint_cov"]:.3f}')
        results['fm_sensitivity'] = sensitivity

        # Also compute joint coverage for FM at N=50 (main result)
        samp_50_cal = fm_cal_samps[:50]
        samp_50_te  = fm_te_samps[:50]
        conf_50 = conformalize_samples(samp_50_cal, tgt_hcal, samp_50_te, tgt_fte, alpha=0.10)
        results['fm_v3b_joint_coverage'] = {
            'joint_cov': conf_50['joint_cov'],
            'avg_marginal_cov': conf_50['avg_marginal_cov'],
            'marginal_cov_per_roi': conf_50['marginal_cov'],
        }
        print(f'\n  FM v3b @ N=50: Joint coverage={conf_50["joint_cov"]:.3f}, '
              f'Marginal={conf_50["avg_marginal_cov"]:.3f}')

    # ────────────────────────────────────────────────────────────────────────
    # 3. Joint coverage for all methods (approximation from Ridge/MLP)
    # ────────────────────────────────────────────────────────────────────────
    print('\n[3/3] Joint coverage for Ridge/MLP (residual bootstrap)...')
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feat_all_tr.numpy())
    X_cal = scaler.transform(feat_hcal.numpy())
    X_te  = scaler.transform(feat_fte.numpy())
    y_tr  = tgt_all_tr.numpy()
    y_cal = tgt_hcal.numpy()
    y_te  = tgt_fte.numpy()

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, y_tr)
    pred_cal = ridge.predict(X_cal)  # [N_cal, 7]
    pred_te  = ridge.predict(X_te)   # [N_te, 7]

    # Generate bootstrap samples for Ridge: add residuals as uncertainty
    res_cal = y_cal - pred_cal  # [N_cal, 7]
    n_boot = 100
    # For each test point, uncertainty = bootstrap of calibration residuals
    ridge_cal_samps = []
    ridge_te_samps  = []
    for _ in range(n_boot):
        idx_b = np.random.randint(0, len(res_cal), size=len(res_cal))
        ridge_cal_samps.append(pred_cal + res_cal[idx_b])
    for _ in range(n_boot):
        idx_b = np.random.randint(0, len(res_cal), size=len(pred_te))
        ridge_te_samps.append(pred_te + res_cal[idx_b])

    ridge_cal_t = torch.tensor(np.array(ridge_cal_samps), dtype=torch.float32)
    ridge_te_t  = torch.tensor(np.array(ridge_te_samps),  dtype=torch.float32)

    conf_ridge = conformalize_samples(ridge_cal_t, tgt_hcal, ridge_te_t, tgt_fte, alpha=0.10)
    results['ridge_joint_coverage'] = {
        'joint_cov': conf_ridge['joint_cov'],
        'avg_marginal_cov': conf_ridge['avg_marginal_cov'],
        'marginal_cov_per_roi': conf_ridge['marginal_cov'],
    }
    print(f'  Ridge: Joint coverage={conf_ridge["joint_cov"]:.3f}, '
          f'Marginal={conf_ridge["avg_marginal_cov"]:.3f}')

    # ────────────────────────────────────────────────────────────────────────
    # Save
    # ────────────────────────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    # ────────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('PHASE C SUMMARY')
    print('='*70)
    print(f'MDN (K=3)  Avg.R={results["mdn_k3"]["avg_r"]:.3f}  '
          f'CRPS={results["mdn_k3"]["avg_crps"]:.3f}  '
          f'CovMAE={results["mdn_k3"]["cov_mae"]:.3f}  '
          f'JointCov={results["mdn_k3"]["joint_cov"]:.3f}')
    if results.get('fm_sensitivity'):
        for ns, v in results['fm_sensitivity'].items():
            print(f'FM N={ns:>3}: R={v["avg_r"]:.3f}  CRPS={v["avg_crps"]:.3f}  '
                  f'CovMAE={v["cov_mae"]:.3f}  JointCov={v["joint_cov"]:.3f}')
    if results.get('fm_v3b_joint_coverage'):
        jc = results['fm_v3b_joint_coverage']
        print(f'FM@50 joint coverage = {jc["joint_cov"]:.3f} (marginal={jc["avg_marginal_cov"]:.3f})')
    if results.get('ridge_joint_coverage'):
        jc = results['ridge_joint_coverage']
        print(f'Ridge joint coverage = {jc["joint_cov"]:.3f} (marginal={jc["avg_marginal_cov"]:.3f})')


if __name__ == '__main__':
    main()
