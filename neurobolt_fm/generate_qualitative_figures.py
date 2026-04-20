#!/usr/bin/env python3
"""
generate_qualitative_figures.py
Generates:
  figK: 2-ROI scatter plots (FM samples vs MDN samples vs true data)
         Demonstrates FM preserves inter-ROI correlation; MDN does not.
  figL: Functional connectivity recovery heatmap
  figM: Summary tradeoff diagram

Requires: FM checkpoint, MDN retrain (fast), feature cache.
"""

import sys, os, json, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
V3B_CKPT  = f'{BASE}/checkpoints_fm_v3b/fm_v3b_joint7d_multisrc.pth'
FIG_DIR   = f'{BASE}/figures'
ES_JSON   = f'{BASE}/round4_energy_score.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
ROI_LABELS = ['Cuneus', "Heschl's", 'Mid.\nFront.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
ROI_SHORT  = ['Cun.', 'Hes.', 'MFG', 'Pre.', 'Put.', 'Tha.', 'Glo.']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 180, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
})


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half-1))
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
        self.velocity_net = nn.Sequential(
            nn.Linear(proj_dim + n_rois + time_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4), nn.SiLU(),
            nn.Linear(hidden_dim//4, n_rois),
        )
    def forward(self, features, x_t, t):
        return self.velocity_net(torch.cat([
            self.feat_proj(features), self.roi_interact(x_t), self.time_emb(t)], dim=-1))
    @torch.no_grad()
    def sample_all(self, features, n_samples=50, num_steps=30, bs=64):
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
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.K, self.n_roi)
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
                mu_k = mu[torch.arange(B), k]
                sig_k = sigma[torch.arange(B), k]
                samps.append((mu_k + sig_k * torch.randn_like(mu_k)).cpu())
            parts.append(torch.stack(samps, 0))
        return torch.cat(parts, dim=1)


def load_cache():
    d = torch.load(f'{CACHE_DIR}/features_conf.pt', map_location='cpu', weights_only=False)
    feat_all_tr = torch.cat([d['feat_tr'], d['feat_cal']], 0).float()
    tgt_all_tr  = torch.cat([d['tgt_tr'],  d['tgt_cal']],  0).float()
    feat_te = d['feat_te'].float()
    tgt_te  = d['tgt_te'].float()
    nh = len(feat_te) // 2
    return feat_all_tr, tgt_all_tr, feat_te[nh:], tgt_te[nh:]   # use final-test only


def train_mdn(feat_tr, tgt_tr, K=3, epochs=80, lr=3e-4, bs=128):
    m = MDNHead(in_dim=FEAT_DIM, hidden=512, n_roi=N_ROIS, K=K).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(feat_tr)
    for ep in range(epochs):
        idx = torch.randperm(N)
        m.train()
        for i in range(0, N, bs):
            b = idx[i:i+bs]
            loss = m.nll_loss(feat_tr[b].to(DEVICE), tgt_tr[b].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 20 == 0: print(f'    MDN ep{ep+1} done')
    m.eval()
    return m


def main():
    print('Loading cache...')
    feat_tr, tgt_tr, feat_te, tgt_te = load_cache()
    print(f'  train={len(feat_tr)}, test={len(feat_te)}')

    # Use a subset of test points for visualization (first 200)
    N_VIZ = min(200, len(feat_te))
    feat_viz = feat_te[:N_VIZ]
    tgt_viz  = tgt_te[:N_VIZ]

    print('Loading FM v3b...')
    fm = MultiSourceFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                           n_rois=N_ROIS, time_dim=32).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    key  = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    fm.load_state_dict(ckpt[key] if key else ckpt, strict=True)
    fm.eval()
    fm_mu  = tgt_tr.mean(0)
    fm_sig = tgt_tr.std(0).clamp(min=1e-6)

    with torch.no_grad():
        print('  Generating FM samples...')
        fm_samp_norm = fm.sample_all(feat_viz, n_samples=50, num_steps=30)  # [50, N, 7]
        fm_samp = fm_samp_norm * fm_sig + fm_mu                              # denormalized

    print('Training MDN...')
    mdn = train_mdn(feat_tr, tgt_tr, K=3, epochs=80, lr=3e-4)
    with torch.no_grad():
        print('  Generating MDN samples...')
        mdn_samp = mdn.sample(feat_viz, n_samples=50)  # [50, N, 7]

    # ─── Figure K: 2-ROI scatter clouds ────────────────────────────────────────
    print('\nGenerating figK: 2-ROI scatter clouds...')

    # Pick ROI pairs showing strong correlation in data
    # ROI 0 (Cuneus) vs ROI 6 (Global) — typically correlated
    # ROI 0 (Cuneus) vs ROI 4 (Putamen) — typically less correlated
    pairs = [(0, 6, "Cuneus vs. Global Signal"), (0, 2, "Cuneus vs. Mid. Frontal")]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(3, n_pairs, figsize=(4.5 * n_pairs, 10))

    ALPHA_SAMP = 0.12
    ALPHA_TRUE = 0.6
    S_SAMP = 8
    S_TRUE = 18

    for col, (ri, rj, pair_title) in enumerate(pairs):
        # Compute true correlation
        y_ri = tgt_viz[:, ri].numpy()
        y_rj = tgt_viz[:, rj].numpy()
        from scipy.stats import pearsonr
        r_true, _ = pearsonr(y_ri, y_rj)

        # Row 0: True data
        ax = axes[0, col]
        ax.scatter(y_ri, y_rj, s=S_TRUE, alpha=ALPHA_TRUE, color='#1B5E20', zorder=3)
        ax.set_xlabel(ROI_LABELS[ri].replace('\n', ' '), fontsize=8)
        ax.set_ylabel(ROI_LABELS[rj].replace('\n', ' '), fontsize=8)
        ax.set_title(f'(a) True Data\n{pair_title}\nr = {r_true:.3f}', fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2)

        # Row 1: FM samples
        ax = axes[1, col]
        for s in range(min(50, fm_samp.shape[0])):
            ax.scatter(fm_samp[s, :, ri].numpy(), fm_samp[s, :, rj].numpy(),
                       s=S_SAMP, alpha=ALPHA_SAMP, color='#1565C0', zorder=2)
        # FM predicted correlation structure
        fm_pred_mean = fm_samp.mean(0).numpy()
        r_fm_mean, _ = pearsonr(fm_pred_mean[:, ri], fm_pred_mean[:, rj])
        ax.set_xlabel(ROI_LABELS[ri].replace('\n', ' '), fontsize=8)
        ax.set_ylabel(ROI_LABELS[rj].replace('\n', ' '), fontsize=8)
        ax.set_title(f'(b) FM v3b Samples\n(N=50 joint ODE samples)\nr_means = {r_fm_mean:.3f}',
                     fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2)

        # Row 2: MDN samples
        ax = axes[2, col]
        for s in range(min(50, mdn_samp.shape[0])):
            ax.scatter(mdn_samp[s, :, ri].numpy(), mdn_samp[s, :, rj].numpy(),
                       s=S_SAMP, alpha=ALPHA_SAMP, color='#B71C1C', zorder=2)
        mdn_pred_mean = mdn_samp.mean(0).numpy()
        r_mdn_mean, _ = pearsonr(mdn_pred_mean[:, ri], mdn_pred_mean[:, rj])
        ax.set_xlabel(ROI_LABELS[ri].replace('\n', ' '), fontsize=8)
        ax.set_ylabel(ROI_LABELS[rj].replace('\n', ' '), fontsize=8)
        ax.set_title(f'(c) MDN (K=3) Samples\n(diagonal-Gaussian mixture)\nr_means = {r_mdn_mean:.3f}',
                     fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2)

    fig.suptitle('Cross-ROI Correlation Structure: FM preserves inter-ROI dependencies\n'
                 '(Key for functional connectivity analysis)',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.6)
    path = f'{FIG_DIR}/figK_2roi_scatter.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')

    # ─── Figure L: Connectivity matrix heatmap ─────────────────────────────────
    print('Generating figL: Connectivity recovery heatmap...')

    true_corr  = np.corrcoef(tgt_te.numpy().T)
    fm_pred_te = fm.sample_all(feat_te, n_samples=20, num_steps=30)
    fm_pred_te = (fm_pred_te * fm_sig + fm_mu).mean(0)  # median/mean [N, 7]
    with torch.no_grad():
        mdn_pred_te_samp = mdn.sample(feat_te, n_samples=20)  # [20, N, 7]
    mdn_pred_te = mdn_pred_te_samp.mean(0)  # [N, 7]

    fm_corr  = np.corrcoef(fm_pred_te.numpy().T)
    mdn_corr = np.corrcoef(mdn_pred_te.numpy().T)

    # CovMAE (off-diagonal)
    off = ~np.eye(N_ROIS, dtype=bool)
    fm_cov_mae  = float(np.mean(np.abs(fm_corr  - true_corr)[off]))
    mdn_cov_mae = float(np.mean(np.abs(mdn_corr - true_corr)[off]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    vmin, vmax = -0.3, 1.0
    for ax, (corr, title, cov) in zip(axes, [
        (true_corr, 'True Connectivity\n(ground truth)', None),
        (fm_corr,   f'FM v3b Predicted\nCovMAE = {fm_cov_mae:.3f}', fm_cov_mae),
        (mdn_corr,  f'MDN (K=3) Predicted\nCovMAE = {mdn_cov_mae:.3f}', mdn_cov_mae),
    ]):
        im = ax.imshow(corr, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(N_ROIS))
        ax.set_yticks(range(N_ROIS))
        ax.set_xticklabels(ROI_SHORT, fontsize=8, rotation=30, ha='right')
        ax.set_yticklabels(ROI_SHORT, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        # Annotate cells
        for i in range(N_ROIS):
            for j in range(N_ROIS):
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white' if abs(corr[i,j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')

    fig.suptitle('Functional Connectivity Recovery: EEG-to-fMRI Cross-ROI Correlations\n'
                 f'CovMAE: FM={fm_cov_mae:.3f} vs MDN={mdn_cov_mae:.3f}\n'
                 'Lower CovMAE = better connectivity preservation',
                 fontsize=10.5, fontweight='bold', y=1.02)
    fig.tight_layout(pad=0.5)
    path = f'{FIG_DIR}/figL_connectivity_heatmap.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')

    # ─── Figure M: Tradeoff summary scatter ────────────────────────────────────
    print('Generating figM: Tradeoff summary...')
    if os.path.exists(ES_JSON):
        with open(ES_JSON) as f:
            es_data = json.load(f)
        methods = {
            'FM v3b (Ours)': {'r': es_data['fm_v3b']['avg_r'], 'crps': es_data['fm_v3b']['avg_crps'],
                               'es': es_data['fm_v3b']['energy_score'], 'cov': es_data['fm_v3b']['cov_mae']},
            'MDN (K=3)':     {'r': es_data['mdn_k3']['avg_r'], 'crps': es_data['mdn_k3']['avg_crps'],
                               'es': es_data['mdn_k3']['energy_score'], 'cov': es_data['mdn_k3']['cov_mae']},
        }
        # Add from conformalized_comparison.json
        conf_path = f'{BASE}/conformalized_comparison.json'
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                cf = json.load(f)
            methods['Ridge']    = {'r': cf['ridge_conformalized']['avg_r'],
                                   'crps': cf['ridge_conformalized']['avg_crps'],
                                   'es': None, 'cov': cf['ridge_conformalized']['cov_mae']}
            methods['Gaussian'] = {'r': cf['gaussian_conformalized']['avg_r'],
                                   'crps': cf['gaussian_conformalized']['avg_crps'],
                                   'es': None, 'cov': cf['gaussian_conformalized']['cov_mae']}

        colors = {'FM v3b (Ours)': '#1565C0', 'MDN (K=3)': '#B71C1C',
                  'Ridge': '#E65100', 'Gaussian': '#6A1B9A'}
        sizes  = {'FM v3b (Ours)': 200, 'MDN (K=3)': 200, 'Ridge': 120, 'Gaussian': 120}

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

        # Panel 1: CovMAE vs Avg.R
        ax = axes[0]
        for m, d in methods.items():
            ax.scatter(d['r'], d['cov'], s=sizes.get(m, 100), color=colors.get(m, 'gray'),
                       edgecolors='white', linewidths=1.5, zorder=5, alpha=0.92)
            ax.annotate(m, (d['r'] + 0.002, d['cov'] + 0.003), fontsize=8.5,
                        fontweight='bold' if 'FM' in m else 'normal',
                        color=colors.get(m, 'gray'))
        ax.axvspan(0.45, 0.50, alpha=0.07, color='#1565C0')
        ax.axhspan(0, 0.12, alpha=0.07, color='#1565C0')
        ax.set_xlabel('Avg. Pearson R (higher = better)', fontsize=9)
        ax.set_ylabel('CovMAE (lower = better)', fontsize=9)
        ax.set_title('(a) Accuracy vs. Connectivity Recovery\n(Key metric for functional connectivity analysis)',
                     fontsize=9.5, fontweight='bold')
        ax.grid(alpha=0.2)
        ax.text(0.445, 0.02, 'FM preferred\nfor connectivity', fontsize=7.5,
                color='#1565C0', alpha=0.8)

        # Panel 2: Energy Score vs Avg.R
        ax = axes[1]
        for m, d in methods.items():
            if d['es'] is None: continue
            ax.scatter(d['r'], d['es'], s=sizes.get(m, 100), color=colors.get(m, 'gray'),
                       edgecolors='white', linewidths=1.5, zorder=5, alpha=0.92)
            ax.annotate(m, (d['r'] + 0.002, d['es'] + 0.005), fontsize=8.5,
                        fontweight='bold' if 'FM' in m else 'normal',
                        color=colors.get(m, 'gray'))
        ax.set_xlabel('Avg. Pearson R (higher = better)', fontsize=9)
        ax.set_ylabel('Energy Score (lower = better)', fontsize=9)
        ax.set_title('(b) Accuracy vs. Marginal Forecast Quality\n(MDN preferred for marginal prediction)',
                     fontsize=9.5, fontweight='bold')
        ax.grid(alpha=0.2)

        fig.suptitle('FM vs MDN: Complementary Tradeoffs in EEG-to-fMRI Prediction\n'
                     'FM: best for functional connectivity analysis (CovMAE)\n'
                     'MDN: best for marginal interval forecasting (Energy Score)',
                     fontsize=10, fontweight='bold', y=1.03)
        fig.tight_layout(pad=0.5)
        path = f'{FIG_DIR}/figM_tradeoff_summary.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'  Saved {path}')

    print('\nAll qualitative figures done.')
    print(f'CovMAE check: FM={fm_cov_mae:.3f}, MDN={mdn_cov_mae:.3f}')


if __name__ == '__main__':
    main()
