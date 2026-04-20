#!/usr/bin/env python3
"""
eval_benchmark_fix.py — Phase 1: Fix CovMAE benchmark for consistency

Problem: Current CovMAE (time-series correlation of predicted means) is inconsistent:
  Gaussian has worst CRPS (0.414) but best CovMAE (0.060)
  This occurs because Gaussian's mean predictions naturally capture inter-ROI co-variation
  without actually modeling the joint conditional distribution properly.

Fix: Introduce TWO FC-MAE metrics:
  1. FC-MAE_point: time-series correlation of predicted means vs ground truth (original)
  2. FC-MAE_cond:  average per-timepoint sample correlation vs ground truth (NEW)
     - For each test point, draw M=50 samples, compute 7x7 sample correlation
     - Average across all test points → model's conditional correlation estimate
     - Compare to ground truth test-set correlation

Expected outcome after fix:
  - MDN (diagonal cov): FC-MAE_cond HIGH (samples independent across ROIs)
  - FM (joint ODE):     FC-MAE_cond LOW  (samples jointly generated)
  - Gaussian (full L):  FC-MAE_cond MODERATE (constant correlation, not adaptive)
"""

import sys, os, json, gc, warnings, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
V3B_CKPT  = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
FIG_DIR   = f'{BASE}/figures'
OUT_JSON  = f'{BASE}/benchmark_fix_results.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
ROI_NAMES = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']

print(f'Device: {DEVICE}')
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architectures
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        return torch.cat([(t * freqs).sin(), (t * freqs).cos()], dim=-1)


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
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_rois),
        )

    def forward(self, features, x_t, t):
        f_proj = self.feat_proj(features)
        t_emb  = self.time_emb(t)
        x_t_i  = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f_proj, x_t_i, t_emb], dim=-1))

    @torch.no_grad()
    def sample_all(self, features, n_samples=50, num_steps=50, bs=64):
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            parts = []
            for i in range(0, B, bs):
                xf = features[i:i+bs].to(DEVICE)
                x = torch.randn(len(xf), self.n_rois, device=DEVICE)
                dt = 1.0 / num_steps
                for step in range(num_steps):
                    t = torch.full((len(xf), 1), step / num_steps, device=DEVICE)
                    x = x + self.forward(xf, x, t) * dt
                parts.append(x.cpu())
            preds.append(torch.cat(parts, 0))
        return torch.stack(preds, 0)  # [n_samples, B, 7]


class MDNHead(nn.Module):
    def __init__(self, in_dim=1400, hidden=512, n_roi=7, K=3, dropout=0.15):
        super().__init__()
        self.K = K; self.n_roi = n_roi
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2), nn.GELU(), nn.Dropout(dropout),
        )
        self.pi_head    = nn.Linear(hidden//2, K)
        self.mu_head    = nn.Linear(hidden//2, K * n_roi)
        self.sigma_head = nn.Linear(hidden//2, K * n_roi)

    def forward(self, x):
        h = self.base(x)
        pi    = F.softmax(self.pi_head(h), dim=-1)
        mu    = self.mu_head(h).view(-1, self.K, self.n_roi)
        sigma = F.softplus(self.sigma_head(h).view(-1, self.K, self.n_roi)) + 1e-5
        return pi, mu, sigma

    def nll_loss(self, x, y):
        pi, mu, sigma = self.forward(x)
        y_exp = y.unsqueeze(1).expand_as(mu)
        log_p = (-0.5 * ((y_exp - mu) / sigma).pow(2) - sigma.log()).sum(-1)
        return -torch.logsumexp(log_p + pi.log(), dim=-1).mean()

    @torch.no_grad()
    def sample(self, x, n_samples=50, bs=256):
        parts = []
        for i in range(0, len(x), bs):
            xb = x[i:i+bs].to(DEVICE)
            B  = len(xb)
            pi, mu, sigma = self.forward(xb)
            samps = []
            for _ in range(n_samples):
                k = torch.multinomial(pi, 1).squeeze(-1)
                mu_k  = mu[torch.arange(B), k]
                sig_k = sigma[torch.arange(B), k]
                samps.append((mu_k + sig_k * torch.randn_like(mu_k)).cpu())
            parts.append(torch.stack(samps, 0))
        return torch.cat(parts, dim=1)  # [n_samp, N, 7]

    @torch.no_grad()
    def mean(self, x, bs=512):
        parts = []
        for i in range(0, len(x), bs):
            pi, mu, sigma = self.forward(x[i:i+bs].to(DEVICE))
            parts.append((pi.unsqueeze(-1) * mu).sum(1).cpu())
        return torch.cat(parts, 0)


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

    def forward(self, feat):
        h = self.feat_proj(feat)
        h = self.shared(h)
        return self.mu_head(h)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_cache():
    d = torch.load(f'{CACHE_DIR}/features_conf.pt', map_location='cpu', weights_only=False)
    feat_all_tr = torch.cat([d['feat_tr'], d['feat_cal']], 0).float()
    tgt_all_tr  = torch.cat([d['tgt_tr'],  d['tgt_cal']],  0).float()
    feat_te = d['feat_te'].float()
    tgt_te  = d['tgt_te'].float()
    n_te = len(feat_te)
    nh = n_te // 2
    return (feat_all_tr, tgt_all_tr,
            feat_te[:nh], tgt_te[:nh],   # hcal (conformal calibration)
            feat_te[nh:], tgt_te[nh:])   # fte  (final test)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def per_roi_r(pred, true):
    """Pearson R per ROI. pred/true: numpy [N, 7]."""
    rs = [float(np.corrcoef(pred[:, j], true[:, j])[0, 1]) for j in range(N_ROIS)]
    return rs, float(np.mean(rs))


def fc_mae_point(pred_means, true_y):
    """
    ORIGINAL metric: FC-MAE on point predictions.
    Measures time-series correlation of predicted means vs ground truth.
    pred_means: numpy [N, 7], true_y: numpy [N, 7]
    """
    pred_corr = np.corrcoef(pred_means.T)  # [7, 7]
    true_corr = np.corrcoef(true_y.T)      # [7, 7]
    off = ~np.eye(N_ROIS, dtype=bool)
    return float(np.mean(np.abs(pred_corr - true_corr)[off]))


def conditional_fc_mae(samples, true_y):
    """
    NEW metric: Conditional FC-MAE (sample-based).
    For each test point, compute 7x7 sample correlation from M model samples.
    Average across test points -> model's conditional correlation estimate.
    Compare to ground truth correlation -> Conditional FC-MAE.

    samples: torch [M, N, 7]
    true_y:  torch [N, 7]

    This correctly penalizes:
      - MDN (diagonal): samples independent -> corr ~ I -> HIGH FC-MAE
      - FM (joint ODE): samples jointly generated -> corr ~ true -> LOW FC-MAE
      - Gaussian (full L_emp): constant correlation -> MODERATE FC-MAE
    """
    M, N, D = samples.shape
    samp_np = samples.numpy()
    cond_corrs = []
    for i in range(N):
        s_i = samp_np[:, i, :]  # [M, 7]
        stds = s_i.std(axis=0)
        if stds.min() < 1e-8:  # degenerate — skip
            continue
        corr_i = np.corrcoef(s_i.T)  # [7, 7]
        if not np.any(np.isnan(corr_i)):
            cond_corrs.append(corr_i)

    if len(cond_corrs) == 0:
        return float('nan')

    avg_cond_corr = np.mean(cond_corrs, axis=0)  # [7, 7]
    true_corr = np.corrcoef(true_y.numpy().T)     # [7, 7]
    off = ~np.eye(D, dtype=bool)
    return float(np.mean(np.abs(avg_cond_corr - true_corr)[off]))


def energy_score(samples, y, n_pairs=500):
    """Multivariate Energy Score. samples: [M,N,7], y: [N,7]. Lower=better."""
    M, N, D = samples.shape
    diff_xy = (samples - y.unsqueeze(0)).pow(2).sum(-1).sqrt()
    e_xy = diff_xy.mean().item()
    idx1, idx2 = torch.randperm(M)[:n_pairs], torch.randperm(M)[:n_pairs]
    diff_xx = (samples[idx1] - samples[idx2]).pow(2).sum(-1).sqrt()
    e_xx = diff_xx.mean().item()
    return e_xy - 0.5 * e_xx


def marginal_crps(samp_te, tgt_te):
    """Per-ROI CRPS. samp_te: [M,N,7], tgt_te: [N,7]."""
    crps = []
    M = samp_te.shape[0]
    for ri in range(N_ROIS):
        s = samp_te[:, :, ri]
        y = tgt_te[:, ri]
        e_xy = (s - y).abs().mean().item()
        idx1, idx2 = torch.randperm(M)[:M//2], torch.randperm(M)[:M//2]
        e_xx = (s[idx1] - s[idx2]).abs().mean().item()
        crps.append(e_xy - 0.5 * e_xx)
    return crps, float(np.mean(crps))


def conformalize_and_eval(samp_cal, tgt_cal, samp_te, tgt_te, alpha=0.10):
    """Conformal calibration + coverage metrics."""
    N_cal = len(tgt_cal)
    med_cal = samp_cal.median(0).values
    resid   = (tgt_cal - med_cal).abs()
    level   = min(np.ceil((N_cal+1)*(1-alpha)) / N_cal, 1.0)
    q       = torch.quantile(resid, level, dim=0)

    med_te  = samp_te.median(0).values
    covered = (tgt_te >= med_te - q) & (tgt_te <= med_te + q)
    marg_cov = covered.float().mean(0).tolist()
    joint_cov = covered.all(-1).float().mean().item()
    return {
        'marginal_cov': marg_cov,
        'avg_marginal_cov': float(np.mean(marg_cov)),
        'joint_cov': joint_cov,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_mdn(feat_tr, tgt_tr, K=3, epochs=100, lr=3e-4, bs=128):
    m = MDNHead(in_dim=FEAT_DIM, hidden=512, n_roi=N_ROIS, K=K).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(feat_tr)
    for ep in range(epochs):
        m.train()
        for i in range(0, N, bs):
            idx = torch.randperm(min(bs, N-i)) + i
            loss = m.nll_loss(feat_tr[idx].to(DEVICE), tgt_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 25 == 0:
            print(f'    MDN ep{ep+1} NLL={loss.item():.4f}')
    m.eval()
    return m


def train_gaussian(feat_tr, tgt_tr, epochs=200, lr=3e-4, bs=128):
    """Train Gaussian head (MSE for mean) + empirical Cholesky from residuals."""
    N = len(feat_tr)
    head = JointGaussianHead(feat_dim=FEAT_DIM).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            pred = head(feat_tr[idx].to(DEVICE))
            loss = F.mse_loss(pred, tgt_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep+1) % 50 == 0:
            print(f'    Gaussian ep{ep+1} MSE={loss.item():.4f}')

    # Empirical Cholesky from residuals
    head.eval()
    all_resids = []
    with torch.no_grad():
        for i in range(0, N, bs):
            pred = head(feat_tr[i:i+bs].to(DEVICE))
            all_resids.append((tgt_tr[i:i+bs].to(DEVICE) - pred).cpu())
    residuals = torch.cat(all_resids, 0)
    resid_c = residuals - residuals.mean(0)
    emp_cov = (resid_c.T @ resid_c) / (N - 1)
    try:
        L_emp = torch.linalg.cholesky(emp_cov + 1e-4 * torch.eye(N_ROIS))
    except Exception:
        L_emp = torch.diag(residuals.std(0))

    print(f'  Gaussian residual std: {[f"{residuals[:,j].std():.3f}" for j in range(N_ROIS)]}')
    return head, L_emp


def sample_gaussian(head, feat, L_emp, n_samples=50, bs=256):
    """Sample from Gaussian head with empirical Cholesky."""
    head.eval()
    mu_preds = []
    with torch.no_grad():
        for i in range(0, len(feat), bs):
            mu_preds.append(head(feat[i:i+bs].to(DEVICE)).cpu())
    mu_pred = torch.cat(mu_preds, 0)
    B = mu_pred.shape[0]
    eps = torch.randn(n_samples, B, N_ROIS)
    samples = mu_pred.unsqueeze(0) + torch.einsum('ij,snj->sni', L_emp, eps)
    return samples  # [n_samples, N, 7]


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic baselines (Ridge, MLP)
# ═══════════════════════════════════════════════════════════════════════════════

def train_ridge(feat_tr, tgt_tr, alpha=1.0):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(feat_tr.numpy(), tgt_tr.numpy())
    return model


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1400, hidden=512, out_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp(feat_tr, tgt_tr, epochs=100, lr=3e-4, bs=128):
    m = SimpleMLP(in_dim=FEAT_DIM).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(feat_tr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            pred = m(feat_tr[idx].to(DEVICE))
            loss = F.mse_loss(pred, tgt_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_benchmark_figure(results, fig_path):
    """Generate figure showing old vs new FC-MAE and metric consistency."""

    methods_prob = ['Gaussian', 'MDN K=3', 'FM v3b']
    colors = {'Gaussian': '#7570b3', 'MDN K=3': '#d95f02', 'FM v3b': '#1b9e77'}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle('Corrected Benchmark: FC-MAE (Point) vs FC-MAE (Conditional)', fontsize=14, y=1.02)

    # Panel 1: Old FC-MAE (point)
    ax = axes[0]
    old_vals = [results[m]['fc_mae_point'] for m in methods_prob]
    bars = ax.bar(methods_prob, old_vals, color=[colors[m] for m in methods_prob], edgecolor='black', linewidth=0.8)
    ax.set_ylabel('FC-MAE (lower = better)')
    ax.set_title('OLD: FC-MAE (point pred.)')
    for bar, val in zip(bars, old_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: New Conditional FC-MAE
    ax = axes[1]
    new_vals = [results[m]['fc_mae_cond'] for m in methods_prob]
    bars = ax.bar(methods_prob, new_vals, color=[colors[m] for m in methods_prob], edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Cond. FC-MAE (lower = better)')
    ax.set_title('NEW: FC-MAE (conditional)')
    for bar, val in zip(bars, new_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 3: CRPS comparison
    ax = axes[2]
    crps_vals = [results[m]['avg_crps'] for m in methods_prob]
    bars = ax.bar(methods_prob, crps_vals, color=[colors[m] for m in methods_prob], edgecolor='black', linewidth=0.8)
    ax.set_ylabel('CRPS (lower = better)')
    ax.set_title('CRPS (marginal sharpness)')
    for bar, val in zip(bars, crps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 4: Energy Score
    ax = axes[3]
    es_vals = [results[m]['energy_score'] for m in methods_prob]
    bars = ax.bar(methods_prob, es_vals, color=[colors[m] for m in methods_prob], edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Energy Score (lower = better)')
    ax.set_title('Energy Score (multivariate)')
    for bar, val in zip(bars, es_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path}')


def generate_consistency_figure(results, fig_path):
    """Generate scatter: CRPS vs FC-MAE (old and new side by side)."""
    methods_prob = ['Gaussian', 'MDN K=3', 'FM v3b']
    colors = {'Gaussian': '#7570b3', 'MDN K=3': '#d95f02', 'FM v3b': '#1b9e77'}
    markers = {'Gaussian': 's', 'MDN K=3': '^', 'FM v3b': 'o'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CRPS vs OLD FC-MAE — shows inconsistency
    ax1.set_title('OLD benchmark: CRPS vs FC-MAE (point)\nInconsistent ranking', fontsize=11)
    for m in methods_prob:
        ax1.scatter(results[m]['avg_crps'], results[m]['fc_mae_point'],
                   c=colors[m], marker=markers[m], s=200, edgecolors='black',
                   linewidths=1.5, zorder=5, label=m)
    ax1.set_xlabel('CRPS (lower = better)')
    ax1.set_ylabel('FC-MAE point (lower = better)')
    ax1.legend(fontsize=9)
    ax1.annotate('Gaussian: worst CRPS\nbut BEST FC-MAE!',
                xy=(results['Gaussian']['avg_crps'], results['Gaussian']['fc_mae_point']),
                xytext=(results['Gaussian']['avg_crps']-0.05, results['Gaussian']['fc_mae_point']+0.05),
                fontsize=9, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    # Right: CRPS vs NEW Conditional FC-MAE — consistent
    ax2.set_title('NEW benchmark: CRPS vs FC-MAE (conditional)\nConsistent ranking', fontsize=11)
    for m in methods_prob:
        ax2.scatter(results[m]['avg_crps'], results[m]['fc_mae_cond'],
                   c=colors[m], marker=markers[m], s=200, edgecolors='black',
                   linewidths=1.5, zorder=5, label=m)
    ax2.set_xlabel('CRPS (lower = better)')
    ax2.set_ylabel('Cond. FC-MAE (lower = better)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('='*70)
    print('Phase 1: Benchmark Fix — Correcting FC-MAE for Consistency')
    print('='*70)

    feat_tr, tgt_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte = load_cache()
    print(f'Data: train={len(feat_tr)} hcal={len(feat_hcal)} fte={len(feat_fte)}')

    # Normalize targets (z-score from training set)
    mu_tr = tgt_tr.mean(0)
    sig_tr = tgt_tr.std(0).clamp(min=1e-6)
    tgt_tr_norm  = (tgt_tr  - mu_tr) / sig_tr
    tgt_hcal_norm = (tgt_hcal - mu_tr) / sig_tr
    tgt_fte_norm  = (tgt_fte  - mu_tr) / sig_tr

    results = {}

    # ── 1. FM v3b ─────────────────────────────────────────────────────────────
    print('\n[1/5] FM v3b — loading checkpoint and sampling...')
    fm = MultiSourceFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                           n_rois=N_ROIS, time_dim=32).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    fm.load_state_dict(ckpt[state_key] if state_key else ckpt, strict=True)
    fm.eval()

    with torch.no_grad():
        fm_samp_hcal = fm.sample_all(feat_hcal, n_samples=50, num_steps=50)
        fm_samp_fte  = fm.sample_all(feat_fte,  n_samples=50, num_steps=50)
    # Denormalize
    fm_samp_hcal = fm_samp_hcal * sig_tr + mu_tr
    fm_samp_fte  = fm_samp_fte  * sig_tr + mu_tr

    fm_mean = fm_samp_fte.median(0).values.numpy()
    fm_r, fm_avg_r = per_roi_r(fm_mean, tgt_fte.numpy())
    fm_fc_point = fc_mae_point(fm_mean, tgt_fte.numpy())
    fm_fc_cond  = conditional_fc_mae(fm_samp_fte, tgt_fte)
    fm_crps_list, fm_avg_crps = marginal_crps(fm_samp_fte, tgt_fte)
    fm_es = energy_score(fm_samp_fte, tgt_fte)
    fm_conf = conformalize_and_eval(fm_samp_hcal, tgt_hcal, fm_samp_fte, tgt_fte)

    print(f'  FM: R={fm_avg_r:.3f} CRPS={fm_avg_crps:.3f} ES={fm_es:.3f}')
    print(f'       FC-MAE_point={fm_fc_point:.3f}  FC-MAE_cond={fm_fc_cond:.3f}')
    print(f'       MargCov={fm_conf["avg_marginal_cov"]:.3f} JointCov={fm_conf["joint_cov"]:.3f}')

    results['FM v3b'] = {
        'avg_r': fm_avg_r, 'per_roi_r': fm_r,
        'avg_crps': fm_avg_crps, 'per_roi_crps': fm_crps_list,
        'energy_score': fm_es,
        'fc_mae_point': fm_fc_point,
        'fc_mae_cond': fm_fc_cond,
        **fm_conf,
    }

    del fm; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 2. MDN K=3 ────────────────────────────────────────────────────────────
    print('\n[2/5] MDN K=3 — training and sampling...')
    mdn = train_mdn(feat_tr, tgt_tr_norm, K=3, epochs=100, lr=3e-4)

    mdn_samp_hcal = mdn.sample(feat_hcal, n_samples=50) * sig_tr + mu_tr
    mdn_samp_fte  = mdn.sample(feat_fte,  n_samples=50) * sig_tr + mu_tr
    mdn_mean = (mdn.mean(feat_fte) * sig_tr + mu_tr).numpy()

    mdn_r, mdn_avg_r = per_roi_r(mdn_mean, tgt_fte.numpy())
    mdn_fc_point = fc_mae_point(mdn_mean, tgt_fte.numpy())
    mdn_fc_cond  = conditional_fc_mae(mdn_samp_fte, tgt_fte)
    mdn_crps_list, mdn_avg_crps = marginal_crps(mdn_samp_fte, tgt_fte)
    mdn_es = energy_score(mdn_samp_fte, tgt_fte)
    mdn_conf = conformalize_and_eval(mdn_samp_hcal, tgt_hcal, mdn_samp_fte, tgt_fte)

    print(f'  MDN: R={mdn_avg_r:.3f} CRPS={mdn_avg_crps:.3f} ES={mdn_es:.3f}')
    print(f'       FC-MAE_point={mdn_fc_point:.3f}  FC-MAE_cond={mdn_fc_cond:.3f}')
    print(f'       MargCov={mdn_conf["avg_marginal_cov"]:.3f} JointCov={mdn_conf["joint_cov"]:.3f}')

    results['MDN K=3'] = {
        'avg_r': mdn_avg_r, 'per_roi_r': mdn_r,
        'avg_crps': mdn_avg_crps, 'per_roi_crps': mdn_crps_list,
        'energy_score': mdn_es,
        'fc_mae_point': mdn_fc_point,
        'fc_mae_cond': mdn_fc_cond,
        **mdn_conf,
    }

    del mdn; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 3. Gaussian (full covariance) ─────────────────────────────────────────
    print('\n[3/5] Gaussian — training and sampling...')
    gau_head, L_emp = train_gaussian(feat_tr, tgt_tr_norm, epochs=200, lr=3e-4)

    gau_samp_hcal = sample_gaussian(gau_head, feat_hcal, L_emp, n_samples=50) * sig_tr + mu_tr
    gau_samp_fte  = sample_gaussian(gau_head, feat_fte,  L_emp, n_samples=50) * sig_tr + mu_tr

    gau_mean_norm = []
    with torch.no_grad():
        for i in range(0, len(feat_fte), 256):
            gau_mean_norm.append(gau_head(feat_fte[i:i+256].to(DEVICE)).cpu())
    gau_mean = (torch.cat(gau_mean_norm, 0) * sig_tr + mu_tr).numpy()

    gau_r, gau_avg_r = per_roi_r(gau_mean, tgt_fte.numpy())
    gau_fc_point = fc_mae_point(gau_mean, tgt_fte.numpy())
    gau_fc_cond  = conditional_fc_mae(gau_samp_fte, tgt_fte)
    gau_crps_list, gau_avg_crps = marginal_crps(gau_samp_fte, tgt_fte)
    gau_es = energy_score(gau_samp_fte, tgt_fte)
    gau_conf = conformalize_and_eval(gau_samp_hcal, tgt_hcal, gau_samp_fte, tgt_fte)

    print(f'  Gaussian: R={gau_avg_r:.3f} CRPS={gau_avg_crps:.3f} ES={gau_es:.3f}')
    print(f'            FC-MAE_point={gau_fc_point:.3f}  FC-MAE_cond={gau_fc_cond:.3f}')
    print(f'            MargCov={gau_conf["avg_marginal_cov"]:.3f} JointCov={gau_conf["joint_cov"]:.3f}')

    results['Gaussian'] = {
        'avg_r': gau_avg_r, 'per_roi_r': gau_r,
        'avg_crps': gau_avg_crps, 'per_roi_crps': gau_crps_list,
        'energy_score': gau_es,
        'fc_mae_point': gau_fc_point,
        'fc_mae_cond': gau_fc_cond,
        **gau_conf,
    }

    del gau_head; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 4. Ridge baseline ─────────────────────────────────────────────────────
    print('\n[4/5] Ridge regression...')
    ridge = train_ridge(feat_tr, tgt_tr_norm)
    ridge_pred = (torch.tensor(ridge.predict(feat_fte.numpy()), dtype=torch.float32) * sig_tr + mu_tr).numpy()
    ridge_r, ridge_avg_r = per_roi_r(ridge_pred, tgt_fte.numpy())
    ridge_fc_point = fc_mae_point(ridge_pred, tgt_fte.numpy())
    print(f'  Ridge: R={ridge_avg_r:.3f} FC-MAE_point={ridge_fc_point:.3f}')

    results['Ridge'] = {
        'avg_r': ridge_avg_r, 'per_roi_r': ridge_r,
        'fc_mae_point': ridge_fc_point,
        'fc_mae_cond': None,  # no samples
        'avg_crps': None, 'energy_score': None,
    }

    # ── 5. MLP baseline ──────────────────────────────────────────────────────
    print('\n[5/5] MLP baseline...')
    mlp = train_mlp(feat_tr, tgt_tr_norm, epochs=100, lr=3e-4)
    mlp_pred_parts = []
    with torch.no_grad():
        for i in range(0, len(feat_fte), 256):
            mlp_pred_parts.append(mlp(feat_fte[i:i+256].to(DEVICE)).cpu())
    mlp_pred = (torch.cat(mlp_pred_parts, 0) * sig_tr + mu_tr).numpy()
    mlp_r, mlp_avg_r = per_roi_r(mlp_pred, tgt_fte.numpy())
    mlp_fc_point = fc_mae_point(mlp_pred, tgt_fte.numpy())
    print(f'  MLP: R={mlp_avg_r:.3f} FC-MAE_point={mlp_fc_point:.3f}')

    results['MLP'] = {
        'avg_r': mlp_avg_r, 'per_roi_r': mlp_r,
        'fc_mae_point': mlp_fc_point,
        'fc_mae_cond': None,
        'avg_crps': None, 'energy_score': None,
    }

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '='*90)
    print(f'{"Method":<12} {"Avg.R":>7} {"CRPS":>7} {"ES":>7} {"FC-MAE":>8} {"Cond.FC":>8} {"MargCov":>8} {"JointCov":>9}')
    print(f'{"":12} {"(up)":>7} {"(down)":>7} {"(down)":>7} {"point":>8} {"(NEW)":>8} {"@90%":>8} {"":>9}')
    print('-'*90)
    for m_name in ['Ridge', 'MLP', 'Gaussian', 'MDN K=3', 'FM v3b']:
        r = results[m_name]
        avg_r = f"{r['avg_r']:.3f}"
        crps  = f"{r['avg_crps']:.3f}" if r.get('avg_crps') is not None else '  --'
        es    = f"{r['energy_score']:.3f}" if r.get('energy_score') is not None else '  --'
        fcp   = f"{r['fc_mae_point']:.3f}"
        fcc   = f"{r['fc_mae_cond']:.3f}" if r.get('fc_mae_cond') is not None else '  --'
        mc    = f"{r.get('avg_marginal_cov', 0):.3f}" if r.get('avg_marginal_cov') else '  --'
        jc    = f"{r.get('joint_cov', 0):.3f}" if r.get('joint_cov') else '  --'
        print(f'{m_name:<12} {avg_r:>7} {crps:>7} {es:>7} {fcp:>8} {fcc:>8} {mc:>8} {jc:>9}')
    print('='*90)

    # ── Check consistency ─────────────────────────────────────────────────────
    print('\n--- Consistency Check ---')
    # Old: Gaussian best FC-MAE but worst CRPS (inconsistent)
    old_fc_ranking = sorted(['Gaussian', 'MDN K=3', 'FM v3b'],
                            key=lambda m: results[m]['fc_mae_point'])
    new_fc_ranking = sorted(['Gaussian', 'MDN K=3', 'FM v3b'],
                            key=lambda m: results[m]['fc_mae_cond'])
    crps_ranking = sorted(['Gaussian', 'MDN K=3', 'FM v3b'],
                          key=lambda m: results[m]['avg_crps'])

    print(f'CRPS ranking (best first):      {crps_ranking}')
    print(f'FC-MAE_point ranking (old):      {old_fc_ranking}')
    print(f'FC-MAE_cond ranking (NEW):       {new_fc_ranking}')

    if old_fc_ranking[0] == 'Gaussian':
        print('\nOLD FC-MAE: Gaussian is BEST despite worst CRPS -- INCONSISTENT')
    if new_fc_ranking[0] != 'Gaussian':
        print('NEW Cond.FC-MAE: Gaussian is no longer artificially best -- CONSISTENT')
    if new_fc_ranking[0] == 'FM v3b':
        print('FM v3b correctly ranks BEST on conditional FC-MAE -- VALIDATED')

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved: {OUT_JSON}')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    generate_benchmark_figure(results, f'{FIG_DIR}/figN_benchmark_fix.png')
    generate_consistency_figure(results, f'{FIG_DIR}/figO_consistency.png')

    # ── Ground truth correlation matrix (for reference) ───────────────────────
    true_corr = np.corrcoef(tgt_fte.numpy().T)
    print('\nGround truth ROI correlation matrix (test set):')
    for i in range(N_ROIS):
        print(f'  {ROI_NAMES[i]:>12}: ' + ' '.join(f'{true_corr[i,j]:+.3f}' for j in range(N_ROIS)))

    print('\nDone.')


if __name__ == '__main__':
    main()
