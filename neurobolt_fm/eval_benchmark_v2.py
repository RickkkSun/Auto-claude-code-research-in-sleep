#!/usr/bin/env python3
"""
eval_benchmark_v2.py — Corrected Probabilistic Benchmark

Metrics:
  1. Avg.R (per-ROI Pearson R) — point prediction accuracy
  2. CRPS — marginal probabilistic quality (per-ROI)
  3. Energy Score — multivariate proper scoring rule
  4. Variogram Score (p=1) — proper scoring rule sensitive to dependence structure
     VS = (1/C(D,2)) sum_{r<s} (|y_r - y_s| - E|X_r - X_s|)^2
  5. FC-MAE (point) — deterministic FC recovery from predicted means
  6. Marginal / Joint Coverage — conformal coverage

Models:
  - Ridge, MLP (deterministic)
  - Gaussian (constant L_emp covariance)
  - HetGaussian (input-dependent Cholesky covariance)
  - MDN K=3 (diagonal mixture)
  - FM v3b (joint 7D ODE)

All metrics include block bootstrap 95% CIs.
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
OUT_JSON  = f'{BASE}/benchmark_v2_results.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
ROI_NAMES = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
print(f'Device: {DEVICE}')
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Architectures
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
                    t_val = torch.full((len(xf), 1), step / num_steps, device=DEVICE)
                    x = x + self.forward(xf, x, t_val) * dt
                parts.append(x.cpu())
            preds.append(torch.cat(parts, 0))
        return torch.stack(preds, 0)


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
            xb = x[i:i+bs].to(DEVICE); B = len(xb)
            pi, mu, sigma = self.forward(xb)
            samps = []
            for _ in range(n_samples):
                k = torch.multinomial(pi, 1).squeeze(-1)
                mu_k  = mu[torch.arange(B), k]
                sig_k = sigma[torch.arange(B), k]
                samps.append((mu_k + sig_k * torch.randn_like(mu_k)).cpu())
            parts.append(torch.stack(samps, 0))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def mean(self, x, bs=512):
        parts = []
        for i in range(0, len(x), bs):
            pi, mu, sigma = self.forward(x[i:i+bs].to(DEVICE))
            parts.append((pi.unsqueeze(-1) * mu).sum(1).cpu())
        return torch.cat(parts, 0)


class GaussianHead(nn.Module):
    """Gaussian with MSE-trained mean + constant empirical Cholesky."""
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7):
        super().__init__()
        self.n_rois = n_rois
        self.net = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_rois),
        )

    def forward(self, feat):
        return self.net(feat)


class HetGaussianHead(nn.Module):
    """Heteroscedastic Gaussian with low-rank + diagonal covariance.
    Sigma(x) = diag(sigma(x)^2) + V(x) V(x)^T
    Sampling: y = mu(x) + diag(sigma(x)) * eps1 + V(x) * eps2
    Much more stable than full Cholesky parameterization.
    """
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7, rank=3):
        super().__init__()
        self.n_rois = n_rois
        self.rank = rank
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, n_rois)
        self.log_diag_head = nn.Linear(hidden_dim, n_rois)
        self.V_head        = nn.Linear(hidden_dim, n_rois * rank)

    def forward(self, feat):
        h = self.feat_proj(feat)
        h = self.shared(h)
        mu = self.mu_head(h)
        log_diag = self.log_diag_head(h).clamp(min=-4, max=0.5)  # sigma in [0.018, 1.65]
        diag = torch.exp(log_diag)  # [B, n_rois]
        V = self.V_head(h).view(-1, self.n_rois, self.rank) * 0.3  # scale down
        return mu, diag, V

    def nll_loss(self, feat, target):
        mu, diag, V = self.forward(feat)
        # Sigma = diag(diag^2) + V V^T
        D = torch.diag_embed(diag.pow(2))  # [B, 7, 7]
        Sigma = D + torch.bmm(V, V.transpose(1, 2))  # [B, 7, 7]
        # Add jitter for numerical stability
        Sigma = Sigma + 1e-4 * torch.eye(self.n_rois, device=feat.device)
        try:
            L = torch.linalg.cholesky(Sigma)
            dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
            return -dist.log_prob(target).mean()
        except Exception:
            # Fallback: just use diagonal part
            dist = torch.distributions.Independent(
                torch.distributions.Normal(mu, diag + 1e-4), 1)
            return -dist.log_prob(target).mean()

    @torch.no_grad()
    def sample_all(self, feat, n_samples=50, bs=256):
        parts = []
        for i in range(0, len(feat), bs):
            fb = feat[i:i+bs].to(DEVICE)
            B = len(fb)
            mu, diag, V = self.forward(fb)
            samps = []
            for _ in range(n_samples):
                eps1 = torch.randn(B, self.n_rois, device=DEVICE)
                eps2 = torch.randn(B, self.rank, device=DEVICE)
                s = mu + diag * eps1 + torch.bmm(V, eps2.unsqueeze(-1)).squeeze(-1)
                samps.append(s.cpu())
            parts.append(torch.stack(samps, 0))  # [n_samp, B, 7]
        return torch.cat(parts, dim=1)  # [n_samp, N, 7]

    @torch.no_grad()
    def predict_mean(self, feat, bs=512):
        parts = []
        for i in range(0, len(feat), bs):
            mu, _, _ = self.forward(feat[i:i+bs].to(DEVICE))
            parts.append(mu.cpu())
        return torch.cat(parts, 0)


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


class ConditionalDenoiser(nn.Module):
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7, time_dim=64):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(time_dim),
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
        # t is [B] integer timesteps — need unsqueeze for SinusoidalEmbedding
        t_inp = t.float().unsqueeze(-1) if t.dim() == 1 else t
        t_emb = self.time_emb(t_inp)
        f_proj = self.feat_proj(features)
        return self.net(torch.cat([f_proj, x_t, t_emb], dim=-1))


class ConditionalDDPM:
    def __init__(self, denoiser, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.denoiser = denoiser; self.T = T; self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = ac
        self.sqrt_ac = torch.sqrt(ac)
        self.sqrt_1mac = torch.sqrt(1.0 - ac)

    def train_loss(self, x_0, features):
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t = self.sqrt_ac[t].unsqueeze(-1) * x_0 + self.sqrt_1mac[t].unsqueeze(-1) * noise
        return F.mse_loss(self.denoiser(x_t, t, features), noise)

    @torch.no_grad()
    def ddim_sample(self, features, n_samples=50, ddim_steps=50, bs=64):
        N = features.shape[0]; nr = self.denoiser.n_rois
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        all_samples = []
        for _ in range(n_samples):
            parts = []
            for i in range(0, N, bs):
                fb = features[i:i+bs].to(self.device); B = len(fb)
                x = torch.randn(B, nr, device=self.device)
                for idx_s in range(len(timesteps)):
                    t_cur = timesteps[idx_s]
                    t_t = torch.full((B,), t_cur, device=self.device, dtype=torch.long)
                    eps = self.denoiser(x, t_t, fb)
                    ab_cur = self.alphas_cumprod[t_cur]
                    ab_prev = self.alphas_cumprod[timesteps[idx_s+1]] if idx_s < len(timesteps)-1 else torch.tensor(1.0, device=self.device)
                    x0 = ((x - torch.sqrt(1-ab_cur)*eps) / torch.sqrt(ab_cur)).clamp(-5, 5)
                    x = torch.sqrt(ab_prev)*x0 + torch.sqrt(1-ab_prev)*eps
                parts.append(x.cpu())
            all_samples.append(torch.cat(parts, 0))
        return torch.stack(all_samples, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Data
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
            feat_te[:nh], tgt_te[:nh],
            feat_te[nh:], tgt_te[nh:])


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def per_roi_r(pred, true):
    rs = [float(np.corrcoef(pred[:, j], true[:, j])[0, 1]) for j in range(N_ROIS)]
    return rs, float(np.mean(rs))


def fc_mae_point(pred_means, true_y):
    """Deterministic FC recovery: correlation of predicted means across time."""
    pred_corr = np.corrcoef(pred_means.T)
    true_corr = np.corrcoef(true_y.T)
    off = ~np.eye(N_ROIS, dtype=bool)
    return float(np.mean(np.abs(pred_corr - true_corr)[off]))


def variogram_score(samples, y, p=1.0):
    """
    Variogram Score (Scheuerer & Hamill, 2015): proper scoring rule sensitive
    to dependence structure.

    VS_p = (1/C(D,2)) * sum_{r<s} (|y_r - y_s|^p - E_F|X_r - X_s|^p)^2

    samples: [M, N, D] — M samples, N test points, D ROIs
    y: [N, D] — observations
    Lower = better. Averaged over test points and ROI pairs.
    """
    M, N, D = samples.shape
    vs_total = 0.0
    n_pairs = 0
    for r in range(D):
        for s in range(r + 1, D):
            obs_diff = (y[:, r] - y[:, s]).abs().pow(p)          # [N]
            samp_diff = (samples[:, :, r] - samples[:, :, s]).abs().pow(p)  # [M, N]
            exp_diff = samp_diff.mean(0)                          # [N]
            vs_total += ((obs_diff - exp_diff) ** 2).mean().item()
            n_pairs += 1
    return vs_total / n_pairs


def energy_score(samples, y):
    """Multivariate Energy Score (vectorized exact). Lower = better.
    samples: [M, N, D], y: [N, D]."""
    M, N, D = samples.shape
    # E||X - y||_2
    diff_xy = (samples - y.unsqueeze(0)).pow(2).sum(-1).sqrt()  # [M, N]
    e_xy = diff_xy.mean().item()
    # E||X - X'||_2 — vectorized upper triangle (M=50 → 1225 pairs)
    idx_i, idx_j = torch.triu_indices(M, M, offset=1)
    diff_xx = (samples[idx_i] - samples[idx_j]).pow(2).sum(-1).sqrt()  # [1225, N]
    e_xx = diff_xx.mean().item()
    return e_xy - 0.5 * e_xx


def marginal_crps(samp_te, tgt_te):
    """Per-ROI CRPS (vectorized exact). samp_te: [M,N,7], tgt_te: [N,7]."""
    crps = []
    M = samp_te.shape[0]
    idx_i, idx_j = torch.triu_indices(M, M, offset=1)
    for ri in range(N_ROIS):
        s = samp_te[:, :, ri]  # [M, N]
        y = tgt_te[:, ri]      # [N]
        e_xy = (s - y).abs().mean().item()
        e_xx = (s[idx_i] - s[idx_j]).abs().mean().item()
        crps.append(e_xy - 0.5 * e_xx)
    return crps, float(np.mean(crps))


def conformalize_and_eval(samp_cal, tgt_cal, samp_te, tgt_te, alpha=0.10):
    N_cal = len(tgt_cal)
    med_cal = samp_cal.median(0).values
    resid   = (tgt_cal - med_cal).abs()
    level   = min(np.ceil((N_cal+1)*(1-alpha)) / N_cal, 1.0)
    q       = torch.quantile(resid, level, dim=0)
    med_te  = samp_te.median(0).values
    covered = (tgt_te >= med_te - q) & (tgt_te <= med_te + q)
    marg_cov = covered.float().mean(0).tolist()
    joint_cov = covered.all(-1).float().mean().item()
    return {'marginal_cov': marg_cov, 'avg_marginal_cov': float(np.mean(marg_cov)),
            'joint_cov': joint_cov}


# ═══════════════════════════════════════════════════════════════════════════════
# Block Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════════════

def block_bootstrap_ci(metric_fn, *args, block_size=29, n_boot=500, alpha=0.05):
    """
    Block bootstrap CI for a metric computed on test data.
    metric_fn(*args) should accept indexed data and return a scalar.
    args should be tensors with N as first dim (or second dim for samples).
    Returns: (point_estimate, ci_lo, ci_hi)
    """
    # Get N from the last arg (tgt_te)
    N = args[-1].shape[0]
    n_blocks = math.ceil(N / block_size)

    point = metric_fn(*args)
    boots = []
    for _ in range(n_boot):
        sel = np.random.randint(0, n_blocks, size=n_blocks)
        idx = []
        for b in sel:
            idx.extend(range(b * block_size, min((b+1)*block_size, N)))
        idx = idx[:N]
        idx_t = torch.tensor(idx)
        # Index each arg
        indexed = []
        for a in args:
            if a.dim() == 2:  # [N, D]
                indexed.append(a[idx_t])
            elif a.dim() == 3:  # [M, N, D]
                indexed.append(a[:, idx_t, :])
            else:
                indexed.append(a)
        boots.append(metric_fn(*indexed))

    boots = np.array(boots)
    return point, float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))


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
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            loss = m.nll_loss(feat_tr[idx].to(DEVICE), tgt_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 25 == 0: print(f'    MDN ep{ep+1} NLL={loss.item():.4f}')
    m.eval(); return m


def train_gaussian(feat_tr, tgt_tr, epochs=200, lr=3e-4, bs=128):
    """Gaussian: MSE-trained mean + constant empirical Cholesky from residuals."""
    N = len(feat_tr)
    head = GaussianHead(feat_dim=FEAT_DIM).to(DEVICE)
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
        if (ep+1) % 50 == 0: print(f'    Gaussian ep{ep+1} MSE={loss.item():.4f}')

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
    return head, L_emp


def sample_gaussian(head, feat, L_emp, n_samples=50, bs=256):
    head.eval()
    mu_preds = []
    with torch.no_grad():
        for i in range(0, len(feat), bs):
            mu_preds.append(head(feat[i:i+bs].to(DEVICE)).cpu())
    mu_pred = torch.cat(mu_preds, 0)
    eps = torch.randn(n_samples, mu_pred.shape[0], N_ROIS)
    return mu_pred.unsqueeze(0) + torch.einsum('ij,snj->sni', L_emp, eps)


def train_het_gaussian(feat_tr, tgt_tr, epochs_mse=150, epochs_nll=50, lr=3e-4, bs=128):
    """
    Heteroscedastic Gaussian: 2-phase training.
    Phase 1: MSE warmup for mean prediction (stable).
    Phase 2: NLL fine-tuning for covariance head only (freeze mean layers).
    """
    N = len(feat_tr)
    head = HetGaussianHead(feat_dim=FEAT_DIM).to(DEVICE)

    # Phase 1: MSE warmup
    print(f'  Phase 1: MSE warmup ({epochs_mse} epochs)...')
    opt1 = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_mse, eta_min=1e-5)
    for ep in range(epochs_mse):
        head.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            mu, _, _ = head(feat_tr[idx].to(DEVICE))
            loss = F.mse_loss(mu, tgt_tr[idx].to(DEVICE))
            opt1.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt1.step()
        sch1.step()
        if (ep+1) % 50 == 0: print(f'    Phase1 ep{ep+1} MSE={loss.item():.4f}')

    # Phase 2: NLL fine-tuning — freeze feat_proj+shared+mu_head, train covariance heads
    print(f'  Phase 2: NLL fine-tune covariance heads ({epochs_nll} epochs)...')
    for p in head.feat_proj.parameters():
        p.requires_grad_(False)
    for p in head.shared.parameters():
        p.requires_grad_(False)
    for p in head.mu_head.parameters():
        p.requires_grad_(False)
    # Only log_diag_head and V_head are trainable

    cov_params = list(head.log_diag_head.parameters()) + list(head.V_head.parameters())
    opt2 = torch.optim.AdamW(cov_params, lr=lr*0.1, weight_decay=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_nll, eta_min=1e-6)
    best_nll = float('inf')
    best_diag_state = None
    best_v_state = None
    for ep in range(epochs_nll):
        head.train()
        perm = torch.randperm(N)
        tot_nll = 0.0; cnt = 0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            loss = head.nll_loss(feat_tr[idx].to(DEVICE), tgt_tr[idx].to(DEVICE))
            if torch.isnan(loss) or loss.item() > 50:
                continue
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(cov_params, 1.0)
            opt2.step()
            tot_nll += loss.item() * len(idx); cnt += len(idx)
        sched_nll = tot_nll / max(cnt, 1)
        if sched_nll < best_nll:
            best_nll = sched_nll
            best_diag_state = {k: v.clone() for k, v in head.log_diag_head.state_dict().items()}
            best_v_state = {k: v.clone() for k, v in head.V_head.state_dict().items()}
        sch2.step()
        if (ep+1) % 10 == 0: print(f'    Phase2 ep{ep+1} NLL={sched_nll:.4f}')

    if best_diag_state is not None:
        head.log_diag_head.load_state_dict(best_diag_state)
        head.V_head.load_state_dict(best_v_state)
        print(f'  Restored best covariance heads (NLL={best_nll:.4f})')

    # Unfreeze all for inference
    for p in head.parameters():
        p.requires_grad_(True)
    head.eval()
    return head


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
            loss = F.mse_loss(m(feat_tr[idx].to(DEVICE)), tgt_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval(); return m


def train_ridge(feat_tr, tgt_tr, alpha=1.0):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(feat_tr.numpy(), tgt_tr.numpy())
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figures(results, prob_methods):
    """Generate benchmark comparison figures."""
    colors = {'Gaussian': '#7570b3', 'HetGaussian': '#e7298a', 'MDN K=3': '#d95f02', 'FM v3b': '#1b9e77'}

    # Figure 1: 4-panel proper scoring rules comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Probabilistic Benchmark: Proper Scoring Rules + FC Recovery', fontsize=13, y=1.02)

    metrics_panels = [
        ('avg_crps', 'CRPS (marginal)', True),
        ('energy_score', 'Energy Score (multivariate)', True),
        ('variogram_score', 'Variogram Score (dependence)', True),
        ('fc_mae_point', 'FC-MAE (deterministic)', True),
    ]
    for ax, (key, title, lower_better) in zip(axes, metrics_panels):
        vals = [results[m][key] for m in prob_methods]
        cols = [colors.get(m, '#999999') for m in prob_methods]
        bars = ax.bar(range(len(prob_methods)), vals, color=cols, edgecolor='black', linewidth=0.8)
        ax.set_xticks(range(len(prob_methods)))
        ax.set_xticklabels(prob_methods, rotation=15, ha='right', fontsize=9)
        ax.set_title(title, fontsize=10)
        direction = 'lower=better' if lower_better else 'higher=better'
        ax.set_ylabel(f'({direction})', fontsize=8)
        # Add CI whiskers if available
        for idx_b, (bar, m) in enumerate(zip(bars, prob_methods)):
            ci = results[m].get(f'{key}_ci')
            if ci:
                ax.errorbar(bar.get_x() + bar.get_width()/2, vals[idx_b],
                           yerr=[[vals[idx_b]-ci[0]], [ci[1]-vals[idx_b]]],
                           fmt='none', ecolor='black', capsize=4, linewidth=1.5)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{vals[idx_b]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figN_benchmark_v2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved figN_benchmark_v2.png')

    # Figure 2: Pairwise scoring comparison scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [('avg_crps', 'variogram_score', 'CRPS vs Variogram Score'),
             ('avg_crps', 'energy_score', 'CRPS vs Energy Score'),
             ('energy_score', 'variogram_score', 'Energy Score vs Variogram Score')]
    markers = {'Gaussian': 's', 'HetGaussian': 'D', 'MDN K=3': '^', 'FM v3b': 'o'}
    for ax, (xk, yk, title) in zip(axes, pairs):
        for m in prob_methods:
            ax.scatter(results[m][xk], results[m][yk],
                      c=colors.get(m, '#999'), marker=markers.get(m, 'o'),
                      s=180, edgecolors='black', linewidths=1.2, zorder=5, label=m)
        ax.set_xlabel(xk.replace('_', ' ').title())
        ax.set_ylabel(yk.replace('_', ' ').title())
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figO_scoring_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved figO_scoring_scatter.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('='*70)
    print('Benchmark v2: Proper Scoring Rules + Variogram Score')
    print('='*70)

    feat_tr, tgt_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte = load_cache()
    print(f'Data: train={len(feat_tr)} hcal={len(feat_hcal)} fte={len(feat_fte)}')

    mu_tr = tgt_tr.mean(0); sig_tr = tgt_tr.std(0).clamp(min=1e-6)
    tgt_tr_norm = (tgt_tr - mu_tr) / sig_tr

    results = {}

    # ── 1. FM v3b ─────────────────────────────────────────────────────────────
    print('\n[1/8] FM v3b...')
    fm = MultiSourceFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                           n_rois=N_ROIS, time_dim=32).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    fm.load_state_dict(ckpt[state_key] if state_key else ckpt, strict=True)
    fm.eval()

    with torch.no_grad():
        fm_samp_hcal = fm.sample_all(feat_hcal, n_samples=50, num_steps=50) * sig_tr + mu_tr
        fm_samp_fte  = fm.sample_all(feat_fte,  n_samples=50, num_steps=50) * sig_tr + mu_tr

    fm_mean = fm_samp_fte.mean(0).numpy()  # use sample mean (consistent with MDN/Gaussian)
    fm_r, fm_avg_r = per_roi_r(fm_mean, tgt_fte.numpy())
    fm_fc = fc_mae_point(fm_mean, tgt_fte.numpy())
    fm_crps_list, fm_avg_crps = marginal_crps(fm_samp_fte, tgt_fte)
    fm_es = energy_score(fm_samp_fte, tgt_fte)
    fm_vs = variogram_score(fm_samp_fte, tgt_fte, p=1.0)
    fm_conf = conformalize_and_eval(fm_samp_hcal, tgt_hcal, fm_samp_fte, tgt_fte)

    # Bootstrap CIs for all metrics
    print('  Computing bootstrap CIs...')
    _, fm_es_lo, fm_es_hi = block_bootstrap_ci(energy_score, fm_samp_fte, tgt_fte, n_boot=300)
    _, fm_vs_lo, fm_vs_hi = block_bootstrap_ci(variogram_score, fm_samp_fte, tgt_fte, n_boot=300)
    def _crps_fn(s, t): return marginal_crps(s, t)[1]
    def _avgr_fn(s, t): _, r = per_roi_r(s.mean(0).numpy(), t.numpy()); return r
    _, fm_crps_lo, fm_crps_hi = block_bootstrap_ci(_crps_fn, fm_samp_fte, tgt_fte, n_boot=300)
    _, fm_r_lo, fm_r_hi = block_bootstrap_ci(_avgr_fn, fm_samp_fte, tgt_fte, n_boot=300)

    print(f'  FM: R={fm_avg_r:.3f}[{fm_r_lo:.3f},{fm_r_hi:.3f}] CRPS={fm_avg_crps:.3f}[{fm_crps_lo:.3f},{fm_crps_hi:.3f}] '
          f'ES={fm_es:.3f}[{fm_es_lo:.3f},{fm_es_hi:.3f}] VS={fm_vs:.4f}[{fm_vs_lo:.4f},{fm_vs_hi:.4f}] FC={fm_fc:.3f}')

    results['FM v3b'] = {
        'avg_r': fm_avg_r, 'avg_r_ci': [fm_r_lo, fm_r_hi], 'per_roi_r': fm_r,
        'avg_crps': fm_avg_crps, 'avg_crps_ci': [fm_crps_lo, fm_crps_hi], 'per_roi_crps': fm_crps_list,
        'energy_score': fm_es, 'energy_score_ci': [fm_es_lo, fm_es_hi],
        'variogram_score': fm_vs, 'variogram_score_ci': [fm_vs_lo, fm_vs_hi],
        'fc_mae_point': fm_fc, **fm_conf,
    }
    sample_store = {'FM v3b': fm_samp_fte}
    del fm; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 2. MDN K=3 ────────────────────────────────────────────────────────────
    print('\n[2/8] MDN K=3...')
    mdn = train_mdn(feat_tr, tgt_tr_norm, K=3, epochs=100, lr=3e-4)
    mdn_samp_hcal = mdn.sample(feat_hcal, n_samples=50) * sig_tr + mu_tr
    mdn_samp_fte  = mdn.sample(feat_fte,  n_samples=50) * sig_tr + mu_tr
    mdn_mean = (mdn.mean(feat_fte) * sig_tr + mu_tr).numpy()

    mdn_r, mdn_avg_r = per_roi_r(mdn_mean, tgt_fte.numpy())
    mdn_fc = fc_mae_point(mdn_mean, tgt_fte.numpy())
    mdn_crps_list, mdn_avg_crps = marginal_crps(mdn_samp_fte, tgt_fte)
    mdn_es = energy_score(mdn_samp_fte, tgt_fte)
    mdn_vs = variogram_score(mdn_samp_fte, tgt_fte, p=1.0)
    mdn_conf = conformalize_and_eval(mdn_samp_hcal, tgt_hcal, mdn_samp_fte, tgt_fte)

    print('  Computing bootstrap CIs...')
    _, mdn_es_lo, mdn_es_hi = block_bootstrap_ci(energy_score, mdn_samp_fte, tgt_fte, n_boot=300)
    _, mdn_vs_lo, mdn_vs_hi = block_bootstrap_ci(variogram_score, mdn_samp_fte, tgt_fte, n_boot=300)
    _, mdn_crps_lo, mdn_crps_hi = block_bootstrap_ci(_crps_fn, mdn_samp_fte, tgt_fte, n_boot=300)
    _, mdn_r_lo, mdn_r_hi = block_bootstrap_ci(_avgr_fn, mdn_samp_fte, tgt_fte, n_boot=300)

    print(f'  MDN: R={mdn_avg_r:.3f}[{mdn_r_lo:.3f},{mdn_r_hi:.3f}] CRPS={mdn_avg_crps:.3f}[{mdn_crps_lo:.3f},{mdn_crps_hi:.3f}] '
          f'ES={mdn_es:.3f}[{mdn_es_lo:.3f},{mdn_es_hi:.3f}] VS={mdn_vs:.4f}[{mdn_vs_lo:.4f},{mdn_vs_hi:.4f}]')

    results['MDN K=3'] = {
        'avg_r': mdn_avg_r, 'avg_r_ci': [mdn_r_lo, mdn_r_hi], 'per_roi_r': mdn_r,
        'avg_crps': mdn_avg_crps, 'avg_crps_ci': [mdn_crps_lo, mdn_crps_hi], 'per_roi_crps': mdn_crps_list,
        'energy_score': mdn_es, 'energy_score_ci': [mdn_es_lo, mdn_es_hi],
        'variogram_score': mdn_vs, 'variogram_score_ci': [mdn_vs_lo, mdn_vs_hi],
        'fc_mae_point': mdn_fc, **mdn_conf,
    }
    sample_store['MDN K=3'] = mdn_samp_fte
    del mdn; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 3. Gaussian (constant L_emp) ──────────────────────────────────────────
    print('\n[3/8] Gaussian (constant covariance)...')
    gau_head, L_emp = train_gaussian(feat_tr, tgt_tr_norm, epochs=200, lr=3e-4)
    gau_samp_hcal = sample_gaussian(gau_head, feat_hcal, L_emp, n_samples=50) * sig_tr + mu_tr
    gau_samp_fte  = sample_gaussian(gau_head, feat_fte,  L_emp, n_samples=50) * sig_tr + mu_tr
    gau_mean_parts = []
    with torch.no_grad():
        for i in range(0, len(feat_fte), 256):
            gau_mean_parts.append(gau_head(feat_fte[i:i+256].to(DEVICE)).cpu())
    gau_mean = (torch.cat(gau_mean_parts, 0) * sig_tr + mu_tr).numpy()

    gau_r, gau_avg_r = per_roi_r(gau_mean, tgt_fte.numpy())
    gau_fc = fc_mae_point(gau_mean, tgt_fte.numpy())
    gau_crps_list, gau_avg_crps = marginal_crps(gau_samp_fte, tgt_fte)
    gau_es = energy_score(gau_samp_fte, tgt_fte)
    gau_vs = variogram_score(gau_samp_fte, tgt_fte, p=1.0)
    gau_conf = conformalize_and_eval(gau_samp_hcal, tgt_hcal, gau_samp_fte, tgt_fte)

    print('  Computing bootstrap CIs...')
    _, gau_es_lo, gau_es_hi = block_bootstrap_ci(energy_score, gau_samp_fte, tgt_fte, n_boot=300)
    _, gau_vs_lo, gau_vs_hi = block_bootstrap_ci(variogram_score, gau_samp_fte, tgt_fte, n_boot=300)
    _, gau_crps_lo, gau_crps_hi = block_bootstrap_ci(_crps_fn, gau_samp_fte, tgt_fte, n_boot=300)
    _, gau_r_lo, gau_r_hi = block_bootstrap_ci(_avgr_fn, gau_samp_fte, tgt_fte, n_boot=300)

    print(f'  Gaussian: R={gau_avg_r:.3f}[{gau_r_lo:.3f},{gau_r_hi:.3f}] CRPS={gau_avg_crps:.3f}[{gau_crps_lo:.3f},{gau_crps_hi:.3f}] '
          f'ES={gau_es:.3f}[{gau_es_lo:.3f},{gau_es_hi:.3f}] VS={gau_vs:.4f}[{gau_vs_lo:.4f},{gau_vs_hi:.4f}]')

    results['Gaussian'] = {
        'avg_r': gau_avg_r, 'avg_r_ci': [gau_r_lo, gau_r_hi], 'per_roi_r': gau_r,
        'avg_crps': gau_avg_crps, 'avg_crps_ci': [gau_crps_lo, gau_crps_hi], 'per_roi_crps': gau_crps_list,
        'energy_score': gau_es, 'energy_score_ci': [gau_es_lo, gau_es_hi],
        'variogram_score': gau_vs, 'variogram_score_ci': [gau_vs_lo, gau_vs_hi],
        'fc_mae_point': gau_fc, **gau_conf,
    }
    sample_store['Gaussian'] = gau_samp_fte
    del gau_head; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 4. HetGaussian (input-dependent covariance) ───────────────────────────
    print('\n[4/8] HetGaussian (input-dependent covariance)...')
    het = train_het_gaussian(feat_tr, tgt_tr_norm, epochs_mse=150, epochs_nll=50, lr=3e-4)
    het_samp_hcal = het.sample_all(feat_hcal, n_samples=50) * sig_tr + mu_tr
    het_samp_fte  = het.sample_all(feat_fte,  n_samples=50) * sig_tr + mu_tr
    het_mean = (het.predict_mean(feat_fte) * sig_tr + mu_tr).numpy()

    het_r, het_avg_r = per_roi_r(het_mean, tgt_fte.numpy())
    het_fc = fc_mae_point(het_mean, tgt_fte.numpy())
    het_crps_list, het_avg_crps = marginal_crps(het_samp_fte, tgt_fte)
    het_es = energy_score(het_samp_fte, tgt_fte)
    het_vs = variogram_score(het_samp_fte, tgt_fte, p=1.0)
    het_conf = conformalize_and_eval(het_samp_hcal, tgt_hcal, het_samp_fte, tgt_fte)

    print('  Computing bootstrap CIs...')
    _, het_es_lo, het_es_hi = block_bootstrap_ci(energy_score, het_samp_fte, tgt_fte, n_boot=300)
    _, het_vs_lo, het_vs_hi = block_bootstrap_ci(variogram_score, het_samp_fte, tgt_fte, n_boot=300)
    _, het_crps_lo, het_crps_hi = block_bootstrap_ci(_crps_fn, het_samp_fte, tgt_fte, n_boot=300)
    _, het_r_lo, het_r_hi = block_bootstrap_ci(_avgr_fn, het_samp_fte, tgt_fte, n_boot=300)

    print(f'  HetGaussian: R={het_avg_r:.3f}[{het_r_lo:.3f},{het_r_hi:.3f}] CRPS={het_avg_crps:.3f}[{het_crps_lo:.3f},{het_crps_hi:.3f}] '
          f'ES={het_es:.3f}[{het_es_lo:.3f},{het_es_hi:.3f}] VS={het_vs:.4f}[{het_vs_lo:.4f},{het_vs_hi:.4f}]')

    results['HetGaussian'] = {
        'avg_r': het_avg_r, 'avg_r_ci': [het_r_lo, het_r_hi], 'per_roi_r': het_r,
        'avg_crps': het_avg_crps, 'avg_crps_ci': [het_crps_lo, het_crps_hi], 'per_roi_crps': het_crps_list,
        'energy_score': het_es, 'energy_score_ci': [het_es_lo, het_es_hi],
        'variogram_score': het_vs, 'variogram_score_ci': [het_vs_lo, het_vs_hi],
        'fc_mae_point': het_fc, **het_conf,
    }
    sample_store['HetGaussian'] = het_samp_fte
    del het; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 5. DDPM ────────────────────────────────────────────────────────────────
    print('\n[5/8] DDPM (DDIM-50)...')
    ddpm_den = ConditionalDenoiser(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                                   n_rois=N_ROIS, time_dim=64).to(DEVICE)
    ddpm = ConditionalDDPM(ddpm_den, T=1000, device=DEVICE)
    ddpm_opt = torch.optim.AdamW(ddpm_den.parameters(), lr=3e-4, weight_decay=1e-4)
    ddpm_sch = torch.optim.lr_scheduler.CosineAnnealingLR(ddpm_opt, T_max=200, eta_min=1e-5)
    N_dd = len(feat_tr)
    for ep in range(200):
        ddpm_den.train()
        perm = torch.randperm(N_dd)
        for i in range(0, N_dd, 128):
            idx = perm[i:i+128]
            loss = ddpm.train_loss(tgt_tr_norm[idx].to(DEVICE), feat_tr[idx].to(DEVICE))
            ddpm_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm_den.parameters(), 1.0)
            ddpm_opt.step()
        ddpm_sch.step()
        if (ep+1) % 50 == 0: print(f'    DDPM ep{ep+1} loss={loss.item():.4f}')
    ddpm_den.eval()

    ddpm_samp_hcal = ddpm.ddim_sample(feat_hcal, n_samples=50, ddim_steps=50) * sig_tr + mu_tr
    ddpm_samp_fte  = ddpm.ddim_sample(feat_fte,  n_samples=50, ddim_steps=50) * sig_tr + mu_tr
    ddpm_mean = ddpm_samp_fte.mean(0).numpy()
    ddpm_r, ddpm_avg_r = per_roi_r(ddpm_mean, tgt_fte.numpy())
    ddpm_fc = fc_mae_point(ddpm_mean, tgt_fte.numpy())
    ddpm_crps_list, ddpm_avg_crps = marginal_crps(ddpm_samp_fte, tgt_fte)
    ddpm_es = energy_score(ddpm_samp_fte, tgt_fte)
    ddpm_vs = variogram_score(ddpm_samp_fte, tgt_fte, p=1.0)
    ddpm_conf = conformalize_and_eval(ddpm_samp_hcal, tgt_hcal, ddpm_samp_fte, tgt_fte)

    print('  Computing bootstrap CIs...')
    _, ddpm_es_lo, ddpm_es_hi = block_bootstrap_ci(energy_score, ddpm_samp_fte, tgt_fte, n_boot=300)
    _, ddpm_vs_lo, ddpm_vs_hi = block_bootstrap_ci(variogram_score, ddpm_samp_fte, tgt_fte, n_boot=300)
    _, ddpm_crps_lo, ddpm_crps_hi = block_bootstrap_ci(_crps_fn, ddpm_samp_fte, tgt_fte, n_boot=300)
    _, ddpm_r_lo, ddpm_r_hi = block_bootstrap_ci(_avgr_fn, ddpm_samp_fte, tgt_fte, n_boot=300)

    print(f'  DDPM: R={ddpm_avg_r:.3f}[{ddpm_r_lo:.3f},{ddpm_r_hi:.3f}] CRPS={ddpm_avg_crps:.3f}[{ddpm_crps_lo:.3f},{ddpm_crps_hi:.3f}] '
          f'ES={ddpm_es:.3f}[{ddpm_es_lo:.3f},{ddpm_es_hi:.3f}] VS={ddpm_vs:.4f}[{ddpm_vs_lo:.4f},{ddpm_vs_hi:.4f}]')

    results['DDPM'] = {
        'avg_r': ddpm_avg_r, 'avg_r_ci': [ddpm_r_lo, ddpm_r_hi], 'per_roi_r': ddpm_r,
        'avg_crps': ddpm_avg_crps, 'avg_crps_ci': [ddpm_crps_lo, ddpm_crps_hi], 'per_roi_crps': ddpm_crps_list,
        'energy_score': ddpm_es, 'energy_score_ci': [ddpm_es_lo, ddpm_es_hi],
        'variogram_score': ddpm_vs, 'variogram_score_ci': [ddpm_vs_lo, ddpm_vs_hi],
        'fc_mae_point': ddpm_fc, **ddpm_conf,
    }
    sample_store['DDPM'] = ddpm_samp_fte
    del ddpm, ddpm_den; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── 6. Ridge ──────────────────────────────────────────────────────────────
    print('\n[6/8] Ridge...')
    ridge = train_ridge(feat_tr, tgt_tr_norm)
    ridge_pred = (torch.tensor(ridge.predict(feat_fte.numpy()), dtype=torch.float32) * sig_tr + mu_tr).numpy()
    ridge_r, ridge_avg_r = per_roi_r(ridge_pred, tgt_fte.numpy())
    ridge_fc = fc_mae_point(ridge_pred, tgt_fte.numpy())
    print(f'  Ridge: R={ridge_avg_r:.3f} FC={ridge_fc:.3f}')
    results['Ridge'] = {'avg_r': ridge_avg_r, 'per_roi_r': ridge_r, 'fc_mae_point': ridge_fc,
                        'avg_crps': None, 'energy_score': None, 'variogram_score': None}
    det_store = {'Ridge': torch.tensor(ridge_pred, dtype=torch.float32)}

    # ── 6. MLP ────────────────────────────────────────────────────────────────
    print('\n[7/8] MLP...')
    mlp = train_mlp(feat_tr, tgt_tr_norm, epochs=100, lr=3e-4)
    mlp_parts = []
    with torch.no_grad():
        for i in range(0, len(feat_fte), 256):
            mlp_parts.append(mlp(feat_fte[i:i+256].to(DEVICE)).cpu())
    mlp_pred = (torch.cat(mlp_parts, 0) * sig_tr + mu_tr).numpy()
    mlp_r, mlp_avg_r = per_roi_r(mlp_pred, tgt_fte.numpy())
    mlp_fc = fc_mae_point(mlp_pred, tgt_fte.numpy())
    print(f'  MLP: R={mlp_avg_r:.3f} FC={mlp_fc:.3f}')
    results['MLP'] = {'avg_r': mlp_avg_r, 'per_roi_r': mlp_r, 'fc_mae_point': mlp_fc,
                      'avg_crps': None, 'energy_score': None, 'variogram_score': None}
    det_store['MLP'] = torch.tensor(mlp_pred, dtype=torch.float32)

    # ═════════════════════════════════════════════════════════════════════════
    # Summary Table
    # ═════════════════════════════════════════════════════════════════════════
    all_methods = ['Ridge', 'MLP', 'Gaussian', 'HetGaussian', 'DDPM', 'MDN K=3', 'FM v3b']
    prob_methods = ['Gaussian', 'HetGaussian', 'DDPM', 'MDN K=3', 'FM v3b']

    print('\n' + '='*100)
    hdr = f'{"Method":<14} {"Avg.R":>7} {"CRPS":>7} {"ES":>10} {"VS":>12} {"FC-MAE":>8} {"MargCov":>8} {"JointCov":>9}'
    print(hdr)
    print(f'{"":14} {"(up)":>7} {"(down)":>7} {"(down)":>10} {"(down)":>12} {"(down)":>8} {"@90%":>8} {"":>9}')
    print('-'*100)
    for m in all_methods:
        r = results[m]
        avg_r = f"{r['avg_r']:.3f}"
        crps  = f"{r['avg_crps']:.3f}" if r.get('avg_crps') is not None else '   --'
        es_ci = r.get('energy_score_ci', [])
        es_str = f"{r['energy_score']:.3f}" if r.get('energy_score') is not None else '   --'
        if es_ci: es_str = f"{r['energy_score']:.3f}+/-{(es_ci[1]-es_ci[0])/2:.3f}"
        vs_ci = r.get('variogram_score_ci', [])
        vs_str = f"{r['variogram_score']:.4f}" if r.get('variogram_score') is not None else '     --'
        if vs_ci: vs_str = f"{r['variogram_score']:.4f}+/-{(vs_ci[1]-vs_ci[0])/2:.4f}"
        fc = f"{r['fc_mae_point']:.3f}"
        mc = f"{r.get('avg_marginal_cov',0):.3f}" if r.get('avg_marginal_cov') else '   --'
        jc = f"{r.get('joint_cov',0):.3f}" if r.get('joint_cov') else '   --'
        print(f'{m:<14} {avg_r:>7} {crps:>7} {es_str:>10} {vs_str:>12} {fc:>8} {mc:>8} {jc:>9}')
    print('='*100)

    # ═════════════════════════════════════════════════════════════════════════
    # Paired Block Bootstrap Significance Tests
    # ═════════════════════════════════════════════════════════════════════════
    print('\n--- Paired Block Bootstrap Significance Tests ---')
    print('  (delta = A - B; for lower-is-better metrics, positive = A worse)')

    def paired_block_bootstrap(metric_fn, samp_a, samp_b, tgt, block_size=29, n_boot=500):
        """Paired bootstrap CI for delta = metric(A) - metric(B)."""
        N = tgt.shape[0]
        n_blocks = math.ceil(N / block_size)
        point_a = metric_fn(samp_a, tgt)
        point_b = metric_fn(samp_b, tgt)
        point_diff = point_a - point_b
        diffs = []
        for _ in range(n_boot):
            sel = np.random.randint(0, n_blocks, size=n_blocks)
            idx = []
            for b_idx in sel:
                idx.extend(range(b_idx * block_size, min((b_idx+1)*block_size, N)))
            idx_t = torch.tensor(idx[:N])
            sa = samp_a[:, idx_t] if samp_a.dim() == 3 else samp_a[idx_t]
            sb = samp_b[:, idx_t] if samp_b.dim() == 3 else samp_b[idx_t]
            tg = tgt[idx_t]
            diffs.append(metric_fn(sa, tg) - metric_fn(sb, tg))
        diffs = np.array(diffs)
        ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        sig = (ci_lo > 0) or (ci_hi < 0)
        return point_diff, ci_lo, ci_hi, sig

    def crps_metric(samp, tgt):
        _, avg = marginal_crps(samp, tgt)
        return avg

    def avg_r_metric(samp, tgt):
        pred = samp.mean(0).numpy()
        _, r = per_roi_r(pred, tgt.numpy())
        return r

    sig_results = {}
    pairs_to_test = [('FM v3b', 'MDN K=3'), ('FM v3b', 'DDPM'),
                     ('FM v3b', 'HetGaussian'), ('FM v3b', 'Gaussian'),
                     ('MDN K=3', 'DDPM'), ('MDN K=3', 'HetGaussian'),
                     ('DDPM', 'HetGaussian'), ('HetGaussian', 'Gaussian')]
    # lower_better: True for ES/VS/CRPS, False for Avg.R
    metrics_to_test = [
        ('Energy Score', energy_score, True),
        ('Variogram Score', variogram_score, True),
        ('CRPS', crps_metric, True),
        ('Avg.R', avg_r_metric, False),
    ]

    for ma, mb in pairs_to_test:
        if ma not in sample_store or mb not in sample_store:
            continue
        sa = sample_store[ma]
        sb = sample_store[mb]
        print(f'\n  {ma} vs {mb}:')
        for m_name, m_fn, lower_better in metrics_to_test:
            diff, ci_lo, ci_hi, sig = paired_block_bootstrap(m_fn, sa, sb, tgt_fte, n_boot=500)
            if not sig:
                winner = 'no sig. difference'
            elif lower_better:
                winner = f'{ma} wins' if diff < 0 else f'{mb} wins'
            else:  # higher is better (Avg.R)
                winner = f'{ma} wins' if diff > 0 else f'{mb} wins'
            print(f'    {m_name:>18}: delta={diff:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] '
                  f'{"SIGNIFICANT" if sig else "not sig."} ({winner})')
            sig_results[f'{ma}_vs_{mb}_{m_name}'] = {
                'diff': diff, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'significant': sig, 'winner': winner
            }

    # FM vs deterministic baselines (Avg.R only — no probabilistic samples for Ridge/MLP)
    def det_avg_r_metric(pred, tgt):
        """Avg.R for deterministic predictions. pred: [N,7], tgt: [N,7]."""
        _, r = per_roi_r(pred.numpy(), tgt.numpy())
        return r

    def paired_det_bootstrap(pred_a, pred_b, tgt, block_size=29, n_boot=500):
        """Paired bootstrap for Avg.R with deterministic predictions."""
        N = tgt.shape[0]
        n_blocks = math.ceil(N / block_size)
        point_a = det_avg_r_metric(pred_a, tgt)
        point_b = det_avg_r_metric(pred_b, tgt)
        point_diff = point_a - point_b
        diffs = []
        for _ in range(n_boot):
            sel = np.random.randint(0, n_blocks, size=n_blocks)
            idx = []
            for b_idx in sel:
                idx.extend(range(b_idx * block_size, min((b_idx+1)*block_size, N)))
            idx_t = torch.tensor(idx[:N])
            diffs.append(det_avg_r_metric(pred_a[idx_t], tgt[idx_t]) -
                        det_avg_r_metric(pred_b[idx_t], tgt[idx_t]))
        diffs = np.array(diffs)
        ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        sig = (ci_lo > 0) or (ci_hi < 0)
        return point_diff, ci_lo, ci_hi, sig

    # FM sample mean vs Ridge/MLP
    fm_det_mean = sample_store['FM v3b'].mean(0)  # [N, 7]
    for det_name in ['Ridge', 'MLP']:
        diff, ci_lo, ci_hi, sig = paired_det_bootstrap(fm_det_mean, det_store[det_name], tgt_fte, n_boot=500)
        if not sig:
            winner = 'no sig. difference'
        else:
            winner = f'FM v3b wins' if diff > 0 else f'{det_name} wins'
        print(f'\n  FM v3b vs {det_name} [Avg.R]: delta={diff:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] '
              f'{"SIGNIFICANT" if sig else "not sig."} ({winner})')
        sig_results[f'FM v3b_vs_{det_name}_Avg.R'] = {
            'diff': diff, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'significant': sig, 'winner': winner
        }

    results['paired_tests'] = sig_results

    # Clean up
    del sample_store, det_store; gc.collect()

    # Save
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved: {OUT_JSON}')

    # Figures
    print('\nGenerating figures...')
    generate_figures(results, prob_methods)
    print('\nDone.')


if __name__ == '__main__':
    main()
