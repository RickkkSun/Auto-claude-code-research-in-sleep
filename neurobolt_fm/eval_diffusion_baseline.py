#!/usr/bin/env python3
"""
eval_diffusion_baseline.py — Conditional DDPM/DDIM baseline for EEG→fMRI

Implements a conditional Denoising Diffusion Probabilistic Model (DDPM)
as a probabilistic baseline on the same NeuroBOLT 1400D features.

Architecture:
  - Condition: NeuroBOLT features f ∈ R^1400 → proj to 512D
  - Noise schedule: linear β from 1e-4 to 0.02, T=1000 steps
  - Denoiser: MLP with time embedding + feature conditioning
  - Inference: DDIM with 50 steps for efficiency

This gives us a third family of probabilistic models:
  Gaussian (parametric) < MDN (mixture) < FM (flow) < DDPM (diffusion)
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
FIG_DIR   = f'{BASE}/figures'
OUT_JSON  = f'{BASE}/diffusion_results.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
print(f'Device: {DEVICE}')


# ═══════════════════════════════════════════════════════════════════════════════
# DDPM Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1))
        emb = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoiser(nn.Module):
    """Conditional denoiser for DDPM. Predicts noise ε given (x_t, t, features)."""
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512, n_rois=7, time_dim=64):
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
        """
        x_t: [B, 7] noisy data
        t: [B] integer timesteps
        features: [B, 1400] conditioning features
        Returns: predicted noise ε [B, 7]
        """
        t_emb = self.time_emb(t)         # [B, time_dim]
        f_proj = self.feat_proj(features)  # [B, proj_dim]
        inp = torch.cat([f_proj, x_t, t_emb], dim=-1)
        return self.net(inp)


class ConditionalDDPM:
    """Conditional DDPM with linear noise schedule and DDIM sampling."""
    def __init__(self, denoiser, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.denoiser = denoiser
        self.T = T
        self.device = device

        # Linear noise schedule
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_1mab * noise, noise

    def train_loss(self, x_0, features):
        """Simple DDPM training loss: E[||ε - ε_θ(x_t, t, f)||^2]"""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        noise_pred = self.denoiser(x_t, t, features)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def ddim_sample(self, features, n_samples=50, ddim_steps=50, eta=0.0, bs=64):
        """DDIM sampling for efficiency. eta=0 → deterministic, eta=1 → DDPM.
        Returns [n_samples, N, 7]."""
        N = features.shape[0]
        # DDIM step indices (evenly spaced from T-1 to 0)
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]  # descending

        all_samples = []
        for _ in range(n_samples):
            parts = []
            for i in range(0, N, bs):
                fb = features[i:i+bs].to(self.device)
                B = len(fb)
                x = torch.randn(B, self.denoiser.n_rois, device=self.device)

                for idx in range(len(timesteps)):
                    t_cur = timesteps[idx]
                    t_tensor = torch.full((B,), t_cur, device=self.device, dtype=torch.long)

                    # Predict noise
                    eps_pred = self.denoiser(x, t_tensor, fb)

                    # Get alpha values
                    ab_cur = self.alphas_cumprod[t_cur]
                    if idx < len(timesteps) - 1:
                        ab_prev = self.alphas_cumprod[timesteps[idx + 1]]
                    else:
                        ab_prev = torch.tensor(1.0, device=self.device)

                    # DDIM update
                    x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_pred) / torch.sqrt(ab_cur)
                    x0_pred = x0_pred.clamp(-5, 5)  # stability clamp

                    sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev))
                    dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * eps_pred
                    noise = sigma * torch.randn_like(x) if sigma > 0 else 0
                    x = torch.sqrt(ab_prev) * x0_pred + dir_xt + noise

                parts.append(x.cpu())
            all_samples.append(torch.cat(parts, 0))
        return torch.stack(all_samples, 0)  # [n_samples, N, 7]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (copied from eval_benchmark_v2.py for consistency)
# ═══════════════════════════════════════════════════════════════════════════════

def per_roi_r(pred, true):
    rs = [float(np.corrcoef(pred[:, j], true[:, j])[0, 1]) for j in range(N_ROIS)]
    return rs, float(np.mean(rs))

def energy_score(samples, y):
    M, N, D = samples.shape
    diff_xy = (samples - y.unsqueeze(0)).pow(2).sum(-1).sqrt()
    e_xy = diff_xy.mean().item()
    idx_i, idx_j = torch.triu_indices(M, M, offset=1)
    diff_xx = (samples[idx_i] - samples[idx_j]).pow(2).sum(-1).sqrt()
    e_xx = diff_xx.mean().item()
    return e_xy - 0.5 * e_xx

def variogram_score(samples, y, p=1.0):
    M, N, D = samples.shape
    vs_total = 0.0; n_pairs = 0
    for r in range(D):
        for s in range(r + 1, D):
            obs_diff = (y[:, r] - y[:, s]).abs().pow(p)
            samp_diff = (samples[:, :, r] - samples[:, :, s]).abs().pow(p)
            exp_diff = samp_diff.mean(0)
            vs_total += ((obs_diff - exp_diff) ** 2).mean().item()
            n_pairs += 1
    return vs_total / n_pairs

def marginal_crps(samp_te, tgt_te):
    crps = []
    M = samp_te.shape[0]
    idx_i, idx_j = torch.triu_indices(M, M, offset=1)
    for ri in range(N_ROIS):
        s = samp_te[:, :, ri]; y = tgt_te[:, ri]
        e_xy = (s - y).abs().mean().item()
        e_xx = (s[idx_i] - s[idx_j]).abs().mean().item()
        crps.append(e_xy - 0.5 * e_xx)
    return crps, float(np.mean(crps))

def fc_mae_point(pred_means, true_y):
    pred_corr = np.corrcoef(pred_means.T)
    true_corr = np.corrcoef(true_y.T)
    off = ~np.eye(N_ROIS, dtype=bool)
    return float(np.mean(np.abs(pred_corr - true_corr)[off]))

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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('='*70)
    print('Conditional DDPM/DDIM Baseline')
    print('='*70)

    feat_tr, tgt_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte = load_cache()
    print(f'Data: train={len(feat_tr)} hcal={len(feat_hcal)} fte={len(feat_fte)}')

    mu_tr = tgt_tr.mean(0); sig_tr = tgt_tr.std(0).clamp(min=1e-6)
    tgt_tr_norm = (tgt_tr - mu_tr) / sig_tr

    # ── Train DDPM ────────────────────────────────────────────────────────────
    print('\n[1/3] Training conditional DDPM...')
    denoiser = ConditionalDenoiser(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                                   n_rois=N_ROIS, time_dim=64).to(DEVICE)
    ddpm = ConditionalDDPM(denoiser, T=1000, device=DEVICE)

    opt = torch.optim.AdamW(denoiser.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
    N = len(feat_tr)
    bs = 128

    for ep in range(200):
        denoiser.train()
        perm = torch.randperm(N)
        tot_loss = 0.0; cnt = 0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            loss = ddpm.train_loss(tgt_tr_norm[idx].to(DEVICE), feat_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * len(idx); cnt += len(idx)
        sched.step()
        if (ep+1) % 25 == 0:
            print(f'    ep{ep+1} loss={tot_loss/cnt:.4f}', flush=True)

    denoiser.eval()

    # ── Sample DDPM (DDIM with 50 steps) ──────────────────────────────────────
    print('\n[2/3] DDIM sampling (50 steps, eta=0)...')
    with torch.no_grad():
        ddpm_samp_hcal = ddpm.ddim_sample(feat_hcal, n_samples=50, ddim_steps=50) * sig_tr + mu_tr
        ddpm_samp_fte  = ddpm.ddim_sample(feat_fte,  n_samples=50, ddim_steps=50) * sig_tr + mu_tr

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('\n[3/3] Evaluating...')
    ddpm_mean = ddpm_samp_fte.mean(0).numpy()
    ddpm_r, ddpm_avg_r = per_roi_r(ddpm_mean, tgt_fte.numpy())
    ddpm_fc = fc_mae_point(ddpm_mean, tgt_fte.numpy())
    ddpm_crps_list, ddpm_avg_crps = marginal_crps(ddpm_samp_fte, tgt_fte)
    ddpm_es = energy_score(ddpm_samp_fte, tgt_fte)
    ddpm_vs = variogram_score(ddpm_samp_fte, tgt_fte, p=1.0)
    ddpm_conf = conformalize_and_eval(ddpm_samp_hcal, tgt_hcal, ddpm_samp_fte, tgt_fte)

    print(f'\n  DDPM: R={ddpm_avg_r:.3f} CRPS={ddpm_avg_crps:.3f} ES={ddpm_es:.3f} '
          f'VS={ddpm_vs:.4f} FC={ddpm_fc:.3f}')
    print(f'        MargCov={ddpm_conf["avg_marginal_cov"]:.3f} JointCov={ddpm_conf["joint_cov"]:.3f}')
    print(f'        Per-ROI R: {[f"{r:.3f}" for r in ddpm_r]}')

    results = {
        'DDPM': {
            'avg_r': ddpm_avg_r, 'per_roi_r': ddpm_r,
            'avg_crps': ddpm_avg_crps, 'per_roi_crps': ddpm_crps_list,
            'energy_score': ddpm_es,
            'variogram_score': ddpm_vs,
            'fc_mae_point': ddpm_fc,
            **ddpm_conf,
        }
    }

    # ── Comparison with known methods ─────────────────────────────────────────
    print('\n' + '='*70)
    print('Comparison (from benchmark_v2_results.json):')
    try:
        with open(f'{BASE}/benchmark_v2_results.json') as f:
            bm = json.load(f)
        print(f'{"Method":<14} {"Avg.R":>7} {"CRPS":>7} {"ES":>7} {"VS":>8}')
        print('-'*50)
        for m in ['Ridge', 'MLP', 'Gaussian', 'HetGaussian', 'MDN K=3', 'FM v3b']:
            if m in bm:
                r = bm[m]
                crps = f"{r['avg_crps']:.3f}" if r.get('avg_crps') else '   --'
                es = f"{r['energy_score']:.3f}" if r.get('energy_score') else '   --'
                vs = f"{r['variogram_score']:.4f}" if r.get('variogram_score') else '    --'
                print(f'{m:<14} {r["avg_r"]:>7.3f} {crps:>7} {es:>7} {vs:>8}')
        print(f'{"DDPM (new)":<14} {ddpm_avg_r:>7.3f} {ddpm_avg_crps:>7.3f} {ddpm_es:>7.3f} {ddpm_vs:>8.4f}')
    except Exception as e:
        print(f'  (Could not load benchmark results: {e})')
    print('='*70)

    # Save
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')


if __name__ == '__main__':
    main()
