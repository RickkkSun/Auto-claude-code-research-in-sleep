#!/usr/bin/env python3
"""
eval_round4_energy_score.py
Computes Energy Score (multivariate proper scoring rule) for FM vs MDN vs Gaussian.
Also computes MDN CovMAE with consistent correlation formula,
and per-scan R with std for table formatting.

Energy Score: ES(F, y) = E_X||X-y||_2 - 0.5*E_{X,X'}||X-X'||_2
Lower = better. Proper multivariate scoring rule — rewards JOINT accuracy.
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
OUT_JSON  = f'{BASE}/round4_energy_score.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
print(f'Device: {DEVICE}')


# ── Architecture (same as eval_conformalized_joint.py) ───────────────────────
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
        return torch.stack(preds, 0)   # [n_samples, B, 7]


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
                k = torch.multinomial(pi, 1).squeeze(-1)     # [B]
                mu_k  = mu[torch.arange(B), k]               # [B, 7]
                sig_k = sigma[torch.arange(B), k]            # [B, 7]
                samps.append((mu_k + sig_k * torch.randn_like(mu_k)).cpu())
            parts.append(torch.stack(samps, 0))  # [n_samp, B, 7]
        return torch.cat(parts, dim=1)  # [n_samp, N_total, 7]

    @torch.no_grad()
    def mean(self, x, bs=512):
        parts = []
        for i in range(0, len(x), bs):
            pi, mu, sigma = self.forward(x[i:i+bs].to(DEVICE))
            parts.append((pi.unsqueeze(-1) * mu).sum(1).cpu())
        return torch.cat(parts, 0)


# ── Load cache ───────────────────────────────────────────────────────────────
def load_cache():
    d = torch.load(f'{CACHE_DIR}/features_conf.pt', map_location='cpu', weights_only=False)
    feat_all_tr = torch.cat([d['feat_tr'], d['feat_cal']], 0).float()
    tgt_all_tr  = torch.cat([d['tgt_tr'],  d['tgt_cal']],  0).float()
    feat_te  = d['feat_te'].float()
    tgt_te   = d['tgt_te'].float()
    n_te = len(feat_te)
    nh   = n_te // 2
    return (feat_all_tr, tgt_all_tr,
            feat_te[:nh], tgt_te[:nh],
            feat_te[nh:], tgt_te[nh:],
            d.get('scan_names', []))


# ── Metrics ──────────────────────────────────────────────────────────────────
def energy_score(samples, y, n_pairs=500):
    """
    Multivariate Energy Score: ES = E||X-y||_2 - 0.5*E||X-X'||_2
    samples: [M, N, 7] — M samples, N test points, 7 ROIs
    y:       [N, 7]
    Lower = better.
    """
    M, N, D = samples.shape
    # E||X - y||_2: average over samples and test points
    # Shape: [M, N] → mean
    diff_xy = (samples - y.unsqueeze(0)).pow(2).sum(-1).sqrt()   # [M, N]
    e_xy = diff_xy.mean().item()
    # E||X - X'||_2: pairs of samples
    idx1 = torch.randperm(M)[:n_pairs]
    idx2 = torch.randperm(M)[:n_pairs]
    diff_xx = (samples[idx1] - samples[idx2]).pow(2).sum(-1).sqrt()  # [n_pairs, N]
    e_xx = diff_xx.mean().item()
    return e_xy - 0.5 * e_xx


def cov_mae_corrcoef(pred_mean, true_y):
    """
    CovMAE using off-diagonal Pearson correlation matrix (consistent with original eval).
    pred_mean: [N, 7]  — point predictions
    true_y:    [N, 7]
    """
    pred_np = pred_mean.numpy()
    true_np = true_y.numpy()
    pred_corr = np.corrcoef(pred_np.T)      # [7, 7]
    true_corr = np.corrcoef(true_np.T)      # [7, 7]
    off = ~np.eye(N_ROIS, dtype=bool)
    return float(np.mean(np.abs(pred_corr - true_corr)[off]))


def per_roi_r(pred, true):
    """Pearson R per ROI."""
    rs = [float(pearsonr(pred[:, ri], true[:, ri])[0]) for ri in range(N_ROIS)]
    return rs, float(np.mean(rs))


def conformalize_and_eval(samp_cal, tgt_cal, samp_te, tgt_te, alpha=0.10):
    """Marginal conformal calibration + metrics."""
    N_cal = len(tgt_cal)
    med_cal = samp_cal.median(0).values
    resid   = (tgt_cal - med_cal).abs()
    level   = min(np.ceil((N_cal+1)*(1-alpha)) / N_cal, 1.0)
    q       = torch.quantile(resid, level, dim=0)   # [7]

    med_te  = samp_te.median(0).values
    covered = (tgt_te >= med_te - q) & (tgt_te <= med_te + q)
    marg_cov = covered.float().mean(0).tolist()
    joint_cov = covered.all(-1).float().mean().item()
    return {
        'marginal_cov': marg_cov,
        'avg_marginal_cov': float(np.mean(marg_cov)),
        'joint_cov': joint_cov,
        'width': float((2*q).mean().item()),
    }


def marginal_crps(samp_te, tgt_te):
    """Per-ROI CRPS (energy score 1D)."""
    crps = []
    M = samp_te.shape[0]
    for ri in range(N_ROIS):
        s = samp_te[:, :, ri]     # [M, N]
        y = tgt_te[:, ri]
        e_xy = (s - y).abs().mean().item()
        idx1, idx2 = torch.randperm(M)[:M//2], torch.randperm(M)[:M//2]
        e_xx = (s[idx1] - s[idx2]).abs().mean().item()
        crps.append(e_xy - 0.5 * e_xx)
    return crps, float(np.mean(crps))


def block_bootstrap_es(samp_a, samp_b, y, block_size=29, n_boot=500):
    """Paired block bootstrap CI for ΔEnergy Score (A - B). Negative = A better."""
    N = y.shape[0]
    n_blocks = math.ceil(N / block_size)
    es_a = energy_score(samp_a, y)
    es_b = energy_score(samp_b, y)
    point = es_a - es_b
    diffs = []
    for _ in range(n_boot):
        sel = np.random.randint(0, n_blocks, size=n_blocks)
        idx = []
        for b in sel:
            idx.extend(range(b * block_size, min((b+1)*block_size, N)))
        idx = torch.tensor(idx[:N])
        ea = energy_score(samp_a[:, idx], y[idx])
        eb = energy_score(samp_b[:, idx], y[idx])
        diffs.append(ea - eb)
    diffs = np.array(diffs)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ── Train MDN ────────────────────────────────────────────────────────────────
def train_mdn(feat_tr, tgt_tr, K=3, epochs=100, lr=3e-4, bs=128):
    m = MDNHead(in_dim=FEAT_DIM, hidden=512, n_roi=N_ROIS, K=K).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(feat_tr)
    for ep in range(epochs):
        idx = torch.randperm(N)
        m.train()
        tot = 0
        for i in range(0, N, bs):
            b = idx[i:i+bs]
            loss = m.nll_loss(feat_tr[b].to(DEVICE), tgt_tr[b].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if (ep+1) % 25 == 0: print(f'    MDN ep{ep+1} NLL={tot/N:.4f}')
    m.eval()
    return m


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    feat_tr, tgt_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte, scan_names = load_cache()
    print(f'Cache: tr={len(feat_tr)} hcal={len(feat_hcal)} fte={len(feat_fte)}')

    results = {}

    # ── FM v3b ────────────────────────────────────────────────────────────────
    print('\n[1/3] Loading and sampling FM v3b...')
    fm = MultiSourceFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                           n_rois=N_ROIS, time_dim=32).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    fm.load_state_dict(ckpt[state_key] if state_key else ckpt, strict=True)
    fm.eval()
    fm_mu  = tgt_tr.mean(0)
    fm_sig = tgt_tr.std(0).clamp(min=1e-6)

    with torch.no_grad():
        print('  Sampling FM on cal...')
        fm_samp_cal_norm = fm.sample_all(feat_hcal, n_samples=50, num_steps=50)
        fm_samp_cal = fm_samp_cal_norm * fm_sig + fm_mu

        print('  Sampling FM on test...')
        fm_samp_te_norm = fm.sample_all(feat_fte, n_samples=50, num_steps=50)
        fm_samp_te = fm_samp_te_norm * fm_sig + fm_mu

    fm_conf = conformalize_and_eval(fm_samp_cal, tgt_hcal, fm_samp_te, tgt_fte)
    fm_r_vals, fm_avg_r = per_roi_r(fm_samp_te.median(0).values.numpy(), tgt_fte.numpy())
    fm_crps_list, fm_avg_crps = marginal_crps(fm_samp_te, tgt_fte)
    fm_es = energy_score(fm_samp_te, tgt_fte)
    fm_cov_mae = cov_mae_corrcoef(fm_samp_te.median(0).values, tgt_fte)

    print(f'  FM: R={fm_avg_r:.3f} CRPS={fm_avg_crps:.3f} ES={fm_es:.4f} '
          f'CovMAE={fm_cov_mae:.3f} JointCov={fm_conf["joint_cov"]:.3f}')

    results['fm_v3b'] = {
        'avg_r': fm_avg_r, 'per_roi_r': fm_r_vals,
        'avg_crps': fm_avg_crps, 'per_roi_crps': fm_crps_list,
        'energy_score': fm_es, 'cov_mae': fm_cov_mae,
        **fm_conf,
    }

    # ── MDN K=3 ───────────────────────────────────────────────────────────────
    print('\n[2/3] Training and sampling MDN (K=3)...')
    mdn = train_mdn(feat_tr, tgt_tr, K=3, epochs=100, lr=3e-4)

    print('  Sampling MDN on cal...')
    mdn_samp_cal = mdn.sample(feat_hcal, n_samples=50)

    print('  Sampling MDN on test...')
    mdn_samp_te  = mdn.sample(feat_fte, n_samples=50)

    mdn_conf = conformalize_and_eval(mdn_samp_cal, tgt_hcal, mdn_samp_te, tgt_fte)
    mdn_mean = mdn.mean(feat_fte)
    mdn_r_vals, mdn_avg_r = per_roi_r(mdn_mean.numpy(), tgt_fte.numpy())
    mdn_crps_list, mdn_avg_crps = marginal_crps(mdn_samp_te, tgt_fte)
    mdn_es = energy_score(mdn_samp_te, tgt_fte)
    mdn_cov_mae = cov_mae_corrcoef(mdn_mean, tgt_fte)

    print(f'  MDN: R={mdn_avg_r:.3f} CRPS={mdn_avg_crps:.3f} ES={mdn_es:.4f} '
          f'CovMAE={mdn_cov_mae:.3f} JointCov={mdn_conf["joint_cov"]:.3f}')

    results['mdn_k3'] = {
        'avg_r': mdn_avg_r, 'per_roi_r': mdn_r_vals,
        'avg_crps': mdn_avg_crps, 'per_roi_crps': mdn_crps_list,
        'energy_score': mdn_es, 'cov_mae': mdn_cov_mae,
        **mdn_conf,
    }

    # ── FM vs MDN statistical comparison ─────────────────────────────────────
    print('\n[3/3] Block bootstrap: FM vs MDN Energy Score...')
    es_diff, es_ci_lo, es_ci_hi = block_bootstrap_es(
        fm_samp_te, mdn_samp_te, tgt_fte, block_size=29, n_boot=500)
    sig = (es_ci_hi < 0) or (es_ci_lo > 0)
    results['fm_vs_mdn_energy_score'] = {
        'diff': es_diff,
        'ci_lo': es_ci_lo,
        'ci_hi': es_ci_hi,
        'significant': bool(sig),
        'fm_better': bool(es_diff < 0),
        'interpretation': (
            f'FM ES={fm_es:.4f}, MDN ES={mdn_es:.4f}. '
            f'Delta(FM-MDN)={es_diff:.4f}, CI=[{es_ci_lo:.4f},{es_ci_hi:.4f}]. '
            f'{"FM significantly better" if sig and es_diff<0 else "MDN significantly better" if sig else "Not significant"}.'
        )
    }
    print(f'  FM ES={fm_es:.4f}, MDN ES={mdn_es:.4f}')
    print(f'  Delta={es_diff:.4f}, CI=[{es_ci_lo:.4f},{es_ci_hi:.4f}]')
    print(f'  Significant: {sig}, FM better: {es_diff < 0}')

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_JSON}')

    print('\n' + '='*70)
    print(f'{"Metric":<20} {"FM v3b":>10} {"MDN K=3":>10} {"FM better?":>12}')
    print('-'*70)
    for metric, key_fm, key_mdn, lower_better in [
        ('Avg.R (↑)',      'avg_r',      'avg_r',      False),
        ('CRPS (↓)',       'avg_crps',   'avg_crps',   True),
        ('Energy Score (↓)', 'energy_score', 'energy_score', True),
        ('CovMAE (↓)',     'cov_mae',    'cov_mae',    True),
        ('Joint Cov (↑)', 'joint_cov',  'joint_cov',  False),
        ('Marg.Cov (↑)',  'avg_marginal_cov', 'avg_marginal_cov', False),
    ]:
        v_fm  = results['fm_v3b'][key_fm]
        v_mdn = results['mdn_k3'][key_mdn]
        fm_better = (v_fm < v_mdn) if lower_better else (v_fm > v_mdn)
        print(f'{metric:<20} {v_fm:>10.3f} {v_mdn:>10.3f} {"FM wins" if fm_better else "MDN wins":>12}')
    print('='*70)
    print(f'Energy Score CI: [{es_ci_lo:.4f}, {es_ci_hi:.4f}] — {"SIGNIFICANT" if sig else "not significant"}')


if __name__ == '__main__':
    main()
