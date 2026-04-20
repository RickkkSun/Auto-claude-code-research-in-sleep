#!/usr/bin/env python3
"""
eval_latent_consistency.py — Cross-Modal Latent Information Retention Probe

Question: Do generative fMRI predictions retain enough information to recover
the original EEG feature representation?

Method:
1. Compress NeuroBOLT 1400D features to compact latent z (64D) via PCA on train set
2. Train a linear probe: real fMRI (7D) → z (64D) on train set only
3. Evaluate on test set:
   - Upper bound: real fMRI → z (how much info fMRI inherently carries about EEG)
   - FM v3b: generated fMRI → z → compare with true z
   - MDN K=3: generated fMRI → z → compare with true z
   - DDPM: generated fMRI → z → compare with true z
4. Report: per-component R, cosine similarity, reconstruction MSE

This is NOT cycle consistency — it's a frozen post-hoc diagnostic that tests
whether generative models preserve cross-modal information.
"""

import sys, os, json, gc, warnings, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
V3B_CKPT  = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
FIG_DIR   = f'{BASE}/figures'
OUT_JSON  = f'{BASE}/latent_consistency_results.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
LATENT_DIM = 64  # PCA components
print(f'Device: {DEVICE}')
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architectures (same as benchmark)
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
        B = features.shape[0]; preds = []
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
    def mean(self, x, bs=512):
        parts = []
        for i in range(0, len(x), bs):
            pi, mu, sigma = self.forward(x[i:i+bs].to(DEVICE))
            parts.append((pi.unsqueeze(-1) * mu).sum(1).cpu())
        return torch.cat(parts, 0)


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
        t_inp = t.float().unsqueeze(-1) if t.dim() == 1 else t
        t_emb = self.time_emb(t_inp)
        f_proj = self.feat_proj(features)
        return self.net(torch.cat([f_proj, x_t, t_emb], dim=-1))


class ConditionalDDPM:
    def __init__(self, denoiser, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.denoiser = denoiser; self.T = T; self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        ac = torch.cumprod(1.0 - betas, dim=0)
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
    def ddim_sample(self, features, n_samples=1, ddim_steps=50, bs=64):
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
# Latent Consistency Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latent_metrics(z_true, z_pred, name):
    """Compute per-component R, cosine sim, MSE."""
    n_comp = z_true.shape[1]
    # Per-component R
    rs = []
    for j in range(n_comp):
        r, _ = pearsonr(z_true[:, j], z_pred[:, j])
        rs.append(r)
    avg_r = float(np.mean(rs))

    # Cosine similarity (per sample, averaged)
    cos_sim = np.mean([np.dot(z_true[i], z_pred[i]) /
                       (np.linalg.norm(z_true[i]) * np.linalg.norm(z_pred[i]) + 1e-8)
                       for i in range(len(z_true))])

    # MSE
    mse = float(np.mean((z_true - z_pred) ** 2))

    print(f'  {name}: Avg latent R={avg_r:.3f}, Cos sim={cos_sim:.3f}, MSE={mse:.4f}')
    return {'avg_latent_r': avg_r, 'cosine_sim': float(cos_sim), 'mse': mse,
            'per_component_r': [float(r) for r in rs[:10]]}  # first 10 components


def main():
    print('='*70)
    print('Cross-Modal Latent Information Retention Probe')
    print('='*70)

    feat_tr, tgt_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte = load_cache()
    print(f'Data: train={len(feat_tr)} hcal={len(feat_hcal)} fte={len(feat_fte)}')

    mu_tr = tgt_tr.mean(0); sig_tr = tgt_tr.std(0).clamp(min=1e-6)
    tgt_tr_norm = (tgt_tr - mu_tr) / sig_tr
    feat_fte_np = feat_fte.numpy()
    feat_tr_np = feat_tr.numpy()

    # ── Step 1: PCA on training features ──────────────────────────────────────
    print(f'\n[Step 1] PCA: {FEAT_DIM}D → {LATENT_DIM}D...')
    pca = PCA(n_components=LATENT_DIM)
    z_tr = pca.fit_transform(feat_tr_np)
    z_te = pca.transform(feat_fte_np)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    print(f'  Variance explained: {var_explained:.3f} ({LATENT_DIM} components)')

    # ── Step 2: Train inverse probe: real fMRI → z on train set ───────────────
    print(f'\n[Step 2] Training inverse probe: fMRI (7D) → z ({LATENT_DIM}D)...')
    # Use denormalized fMRI for probe (what the generative models output)
    probe = Ridge(alpha=10.0)
    probe.fit(tgt_tr.numpy(), z_tr)

    # Upper bound: real fMRI → z on test set
    z_pred_real = probe.predict(tgt_fte.numpy())
    real_metrics = compute_latent_metrics(z_te, z_pred_real, 'Real fMRI (upper bound)')

    # ── Step 3: Generate fMRI from each model ─────────────────────────────────
    results = {'pca_variance_explained': var_explained}
    results['real_fmri'] = real_metrics

    # Trivial baseline: predict z from train-set mean fMRI
    mean_fmri = tgt_tr.mean(0).numpy()
    z_pred_trivial = probe.predict(np.tile(mean_fmri, (len(feat_fte), 1)))
    trivial_metrics = compute_latent_metrics(z_te, z_pred_trivial, 'Trivial (mean fMRI)')
    results['trivial'] = trivial_metrics

    # ── FM v3b ────────────────────────────────────────────────────────────────
    print(f'\n[Step 3a] FM v3b...')
    fm = MultiSourceFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                           n_rois=N_ROIS, time_dim=32).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    fm.load_state_dict(ckpt[state_key] if state_key else ckpt, strict=True)
    fm.eval()
    with torch.no_grad():
        fm_samp = fm.sample_all(feat_fte, n_samples=10, num_steps=50) * sig_tr + mu_tr
    fm_mean = fm_samp.mean(0).numpy()
    z_pred_fm = probe.predict(fm_mean)
    fm_metrics = compute_latent_metrics(z_te, z_pred_fm, 'FM v3b')
    results['fm_v3b'] = fm_metrics
    del fm; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── MDN K=3 ───────────────────────────────────────────────────────────────
    print(f'\n[Step 3b] MDN K=3...')
    mdn = MDNHead(in_dim=FEAT_DIM, hidden=512, n_roi=N_ROIS, K=3).to(DEVICE)
    mdn_opt = torch.optim.AdamW(mdn.parameters(), lr=3e-4, weight_decay=1e-4)
    mdn_sch = torch.optim.lr_scheduler.CosineAnnealingLR(mdn_opt, T_max=100)
    N = len(feat_tr)
    for ep in range(100):
        mdn.train()
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            loss = mdn.nll_loss(feat_tr[idx].to(DEVICE), tgt_tr_norm[idx].to(DEVICE))
            mdn_opt.zero_grad(); loss.backward(); mdn_opt.step()
        mdn_sch.step()
    mdn.eval()
    mdn_mean = (mdn.mean(feat_fte) * sig_tr + mu_tr).numpy()
    z_pred_mdn = probe.predict(mdn_mean)
    mdn_metrics = compute_latent_metrics(z_te, z_pred_mdn, 'MDN K=3')
    results['mdn_k3'] = mdn_metrics
    del mdn; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── DDPM ──────────────────────────────────────────────────────────────────
    print(f'\n[Step 3c] DDPM...')
    ddpm_den = ConditionalDenoiser(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                                   n_rois=N_ROIS, time_dim=64).to(DEVICE)
    ddpm = ConditionalDDPM(ddpm_den, T=1000, device=DEVICE)
    ddpm_opt = torch.optim.AdamW(ddpm_den.parameters(), lr=3e-4, weight_decay=1e-4)
    ddpm_sch = torch.optim.lr_scheduler.CosineAnnealingLR(ddpm_opt, T_max=200, eta_min=1e-5)
    for ep in range(200):
        ddpm_den.train()
        perm = torch.randperm(N)
        for i in range(0, N, 128):
            idx = perm[i:i+128]
            loss = ddpm.train_loss(tgt_tr_norm[idx].to(DEVICE), feat_tr[idx].to(DEVICE))
            ddpm_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm_den.parameters(), 1.0)
            ddpm_opt.step()
        ddpm_sch.step()
        if (ep+1) % 50 == 0: print(f'    DDPM ep{ep+1}')
    ddpm_den.eval()
    ddpm_samp = ddpm.ddim_sample(feat_fte, n_samples=10, ddim_steps=50) * sig_tr + mu_tr
    ddpm_mean = ddpm_samp.mean(0).numpy()
    z_pred_ddpm = probe.predict(ddpm_mean)
    ddpm_metrics = compute_latent_metrics(z_te, z_pred_ddpm, 'DDPM')
    results['ddpm'] = ddpm_metrics
    del ddpm, ddpm_den; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print(f'{"Source":<25} {"Latent R":>10} {"Cos Sim":>10} {"MSE":>10}')
    print('-'*70)
    for name, key in [('Trivial (mean fMRI)', 'trivial'),
                      ('Real fMRI (upper bound)', 'real_fmri'),
                      ('FM v3b', 'fm_v3b'),
                      ('MDN K=3', 'mdn_k3'),
                      ('DDPM', 'ddpm')]:
        r = results[key]
        print(f'{name:<25} {r["avg_latent_r"]:>10.3f} {r["cosine_sim"]:>10.3f} {r["mse"]:>10.4f}')
    print('='*70)
    print(f'PCA variance explained: {var_explained:.3f}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cross-Modal Latent Information Retention', fontsize=13)

    methods = ['Trivial', 'Real fMRI', 'FM v3b', 'MDN K=3', 'DDPM']
    keys = ['trivial', 'real_fmri', 'fm_v3b', 'mdn_k3', 'ddpm']
    colors = ['#999999', '#333333', '#1b9e77', '#d95f02', '#7570b3']

    # Panel 1: Latent R
    lat_rs = [results[k]['avg_latent_r'] for k in keys]
    bars = ax1.bar(methods, lat_rs, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Avg Latent R (higher = better)')
    ax1.set_title('fMRI → Latent EEG Feature Recovery')
    for bar, val in zip(bars, lat_rs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticklabels(methods, rotation=15, ha='right')

    # Panel 2: Cosine similarity
    cos_sims = [results[k]['cosine_sim'] for k in keys]
    bars = ax2.bar(methods, cos_sims, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Cosine Similarity (higher = better)')
    ax2.set_title('fMRI → Latent EEG Feature Direction')
    for bar, val in zip(bars, cos_sims):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticklabels(methods, rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/figP_latent_consistency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {FIG_DIR}/figP_latent_consistency.png')

    # Save
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {OUT_JSON}')
    print('\nDone.')


if __name__ == '__main__':
    main()
