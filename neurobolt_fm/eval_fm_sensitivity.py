#!/usr/bin/env python3
"""
eval_fm_sensitivity.py
FM v3b sample sensitivity + joint coverage analysis.
Uses correct MultiSourceFMHead architecture from eval_conformalized_joint.py.
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
PHASEC_JSON = f'{BASE}/phase_c_results.json'

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ROIS   = 7
FEAT_DIM = 1400
print(f'Device: {DEVICE}')


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half-1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


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
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4), nn.SiLU(),
            nn.Linear(hidden_dim//4, n_rois),
        )

    def forward(self, features, x_t, t):
        f_proj = self.feat_proj(features)
        t_emb  = self.time_emb(t)
        x_t_i  = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f_proj, x_t_i, t_emb], dim=-1))

    @torch.no_grad()
    def sample_all(self, features, n_samples=50, num_steps=100, bs=64):
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x_parts = []
            for i in range(0, B, bs):
                xf = features[i:i+bs].to(DEVICE)
                x = torch.randn(len(xf), self.n_rois, device=DEVICE)
                dt = 1.0 / num_steps
                for step in range(num_steps):
                    t = torch.full((len(xf), 1), step / num_steps, device=DEVICE)
                    x = x + self.forward(xf, x, t) * dt
                x_parts.append(x.cpu())
            preds.append(torch.cat(x_parts, 0))
        return torch.stack(preds, 0)  # [n_samples, B, 7]


def load_cache():
    d = torch.load(f'{CACHE_DIR}/features_conf.pt', map_location='cpu', weights_only=False)
    feat_all_tr = torch.cat([d['feat_tr'], d['feat_cal']], 0).float()
    tgt_all_tr  = torch.cat([d['tgt_tr'],  d['tgt_cal']],  0).float()
    feat_te = d['feat_te'].float()
    tgt_te  = d['tgt_te'].float()
    n_te = len(feat_te)
    nh   = n_te // 2
    return (feat_all_tr, tgt_all_tr,
            feat_te[:nh], tgt_te[:nh],
            feat_te[nh:], tgt_te[nh:])


def conformalize(samp_cal, tgt_cal, samp_te, tgt_te, alpha=0.10):
    """Marginal split conformal; also compute joint coverage."""
    N_cal = len(tgt_cal)
    med_cal = samp_cal.median(0).values
    resid   = (tgt_cal - med_cal).abs()
    level = min(np.ceil((N_cal+1)*(1-alpha)) / N_cal, 1.0)
    q = torch.quantile(resid, level, dim=0)      # [7]
    med_te = samp_te.median(0).values
    covered = (tgt_te >= med_te - q) & (tgt_te <= med_te + q)  # [N_te, 7]
    marg_cov = covered.float().mean(0).tolist()
    joint_cov = covered.all(-1).float().mean().item()
    return {
        'q': q.tolist(),
        'marginal_cov': marg_cov,
        'avg_marginal_cov': float(np.mean(marg_cov)),
        'joint_cov': joint_cov,
        'width': float((2*q).mean().item()),
    }


def crps_from_samples(samp_te, tgt_te):
    """Energy-based CRPS per ROI."""
    crps = []
    for ri in range(N_ROIS):
        s = samp_te[:, :, ri]          # [M, N]
        y = tgt_te[:, ri]
        e_xy = (s - y).abs().mean().item()
        M = s.shape[0]
        idx1 = torch.randperm(M)[:M//2]
        idx2 = torch.randperm(M)[:M//2]
        e_xx = (s[idx1] - s[idx2]).abs().mean().item()
        crps.append(e_xy - 0.5 * e_xx)
    return crps, float(np.mean(crps))


def cov_mae(samp_te, tgt_te):
    pred_mean = samp_te.mean(0)
    pred_cov = torch.cov(pred_mean.T)
    true_cov = torch.cov(tgt_te.T)
    return float((pred_cov - true_cov).abs().mean().item())


def r_vals(samp_te, tgt_te):
    pred = samp_te.median(0).values.numpy()
    y    = tgt_te.numpy()
    rs   = [float(pearsonr(pred[:,ri], y[:,ri])[0]) for ri in range(N_ROIS)]
    return rs, float(np.mean(rs))


def main():
    print('Loading cache...')
    feat_tr, tgt_tr, feat_hcal, tgt_hcal, feat_fte, tgt_fte = load_cache()
    print(f'  train={len(feat_tr)}, hcal={len(feat_hcal)}, fte={len(feat_fte)}')

    print('Loading FM v3b...')
    fm = MultiSourceFMHead(feat_dim=FEAT_DIM, proj_dim=512, hidden_dim=512,
                           n_rois=N_ROIS, time_dim=32).to(DEVICE)
    ckpt = torch.load(V3B_CKPT, map_location='cpu', weights_only=False)
    state_key = 'head' if 'head' in ckpt else ('model' if 'model' in ckpt else None)
    state = ckpt[state_key] if state_key else ckpt
    fm.load_state_dict(state, strict=True)
    fm.eval()
    print('  FM v3b loaded OK.')

    # FM normalization
    fm_mu  = tgt_tr.mean(0)
    fm_sig = tgt_tr.std(0).clamp(min=1e-6)

    print('\nGenerating 100 samples on cal set...')
    with torch.no_grad():
        # Normalize features are already extracted; normalize targets
        # FM was trained on normalized targets
        samp_cal_norm = fm.sample_all(feat_hcal, n_samples=100, num_steps=50)  # [100, N_cal, 7]
        # Denormalize
        samp_cal = samp_cal_norm * fm_sig + fm_mu   # [100, N_cal, 7]

    print('Generating 100 samples on test set...')
    with torch.no_grad():
        samp_te_norm = fm.sample_all(feat_fte, n_samples=100, num_steps=50)    # [100, N_te, 7]
        samp_te = samp_te_norm * fm_sig + fm_mu                                # [100, N_te, 7]

    print('Computing sensitivity...')
    sens = {}
    for n_samp in [10, 20, 50, 100]:
        sc = samp_cal[:n_samp]
        st = samp_te[:n_samp]
        conf = conformalize(sc, tgt_hcal, st, tgt_fte, alpha=0.10)
        rs, avg_r = r_vals(st, tgt_fte)
        crps_list, avg_crps = crps_from_samples(st, tgt_fte)
        cm = cov_mae(st, tgt_fte)
        sens[str(n_samp)] = {
            'avg_r': avg_r, 'per_roi_r': rs,
            'avg_crps': avg_crps, 'per_roi_crps': crps_list,
            'cov_mae': cm, **conf,
        }
        print(f'  N={n_samp:3d}: R={avg_r:.3f}  CRPS={avg_crps:.3f}  '
              f'CovMAE={cm:.3f}  MargCov={conf["avg_marginal_cov"]:.3f}  '
              f'JointCov={conf["joint_cov"]:.3f}')

    # Load and update phase_c_results.json
    if os.path.exists(PHASEC_JSON):
        with open(PHASEC_JSON) as f:
            phasec = json.load(f)
    else:
        phasec = {}

    phasec['fm_sensitivity'] = sens
    fm50 = sens['50']
    phasec['fm_v3b_joint_coverage'] = {
        'joint_cov': fm50['joint_cov'],
        'avg_marginal_cov': fm50['avg_marginal_cov'],
        'marginal_cov_per_roi': fm50['marginal_cov'],
    }

    with open(PHASEC_JSON, 'w') as f:
        json.dump(phasec, f, indent=2)
    print(f'\nUpdated {PHASEC_JSON}')

    print('\n' + '='*65)
    print('SAMPLE SENSITIVITY SUMMARY')
    print(f'{"N":>5} {"Avg.R":>8} {"CRPS":>8} {"CovMAE":>8} {"MargCov":>8} {"JointCov":>10}')
    print('-'*65)
    for ns in [10, 20, 50, 100]:
        v = sens[str(ns)]
        print(f'{ns:>5} {v["avg_r"]:>8.3f} {v["avg_crps"]:>8.3f} {v["cov_mae"]:>8.3f} '
              f'{v["avg_marginal_cov"]:>8.3f} {v["joint_cov"]:>10.3f}')


if __name__ == '__main__':
    main()
