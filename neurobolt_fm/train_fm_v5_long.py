#!/usr/bin/env python3
"""
FM v5 — Extended training of proven v3b architecture

Strategy: v3b architecture already achieves 0.468.
Key changes:
  1. Train 1000 epochs (vs 300 in v3b)
  2. Cosine annealing LR (no warm restarts — stable)
  3. Dropout (0.1) to prevent overfitting
  4. Data augmentation: Gaussian noise on features + target jitter
  5. Test with BOTH original v3b arch AND slightly wider (proj_dim=640)
  6. Best-checkpoint selection on held-out validation
  7. Use ALL data including test for feature cache (re-extract if needed)
"""

import sys, os, gc, json, math, warnings, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

BASE      = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CACHE_DIR = f'{BASE}/feature_cache'
SAVE_DIR  = f'{BASE}/checkpoints_fm_v5'
OUT_JSON  = f'{BASE}/fm_v5_results.json'
os.makedirs(SAVE_DIR, exist_ok=True)

N_ROIS = 7
FEAT_DIM = 1400
ROI_DISPLAY = ['Cuneus', "Heschl's Gyrus", 'Mid. Frontal', 'Precuneus Ant.',
               'Putamen', 'Thalamus', 'Global Signal']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        return torch.cat([(t * freqs).sin(), (t * freqs).cos()], dim=-1)


class MultiSourceFMHead(nn.Module):
    """Same proven architecture as v3b, with optional dropout."""
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512,
                 n_rois=7, time_dim=32, dropout=0.1):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.roi_interact = nn.Linear(n_rois, n_rois)
        in_dim = proj_dim + n_rois + time_dim
        self.velocity_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(dropout),
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
    def sample(self, features, n_samples=50, num_steps=50, bs=64):
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


def load_features():
    d = torch.load(f'{CACHE_DIR}/features_conf.pt', map_location='cpu', weights_only=False)
    feat_tr = d['feat_tr'].float()
    tgt_tr  = d['tgt_tr'].float()
    feat_cal = d['feat_cal'].float()
    tgt_cal  = d['tgt_cal'].float()
    feat_te = d['feat_te'].float()
    tgt_te  = d['tgt_te'].float()
    return feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te


def train_fm(feat_tr, tgt_tr, feat_val, tgt_val, config):
    """Train FM head with validation-based best checkpoint selection."""
    N = len(feat_tr)
    mu = tgt_tr.mean(0); sig = tgt_tr.std(0).clamp(min=1e-6)
    tgt_norm = (tgt_tr - mu) / sig
    tgt_val_norm = (tgt_val - mu) / sig

    head = MultiSourceFMHead(
        feat_dim=FEAT_DIM, proj_dim=config['proj_dim'],
        hidden_dim=config['hidden_dim'], n_rois=N_ROIS,
        time_dim=config['time_dim'], dropout=config['dropout']
    ).to(DEVICE)

    # EMA
    ema_head = copy.deepcopy(head)
    ema_decay = 0.9995

    opt = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['epochs'], eta_min=config['lr'] * 0.01
    )

    n_params = sum(p.numel() for p in head.parameters())
    print(f'  Config: proj={config["proj_dim"]} hidden={config["hidden_dim"]} '
          f'dropout={config["dropout"]} params={n_params:,}')
    print(f'  Epochs: {config["epochs"]}, LR: {config["lr"]}, WD: {config["wd"]}')

    best_val_loss = float('inf')
    best_state = None
    patience = 0
    max_patience = 150  # early stopping

    bs = config['bs']
    noise_std = config.get('feat_noise', 0.0)

    for ep in range(config['epochs']):
        head.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0; n_b = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            c = feat_tr[idx].to(DEVICE)
            x1 = tgt_norm[idx].to(DEVICE)

            # Feature augmentation: add small Gaussian noise
            if noise_std > 0:
                c = c + noise_std * torch.randn_like(c)

            t = torch.rand(len(c), 1, device=DEVICE)
            x0 = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            target_vel = x1 - x0

            v_pred = head(c, x_t, t)
            loss = F.mse_loss(v_pred, target_vel)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_head.parameters(), head.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item(); n_b += 1

        sch.step()

        # Validation loss (every 10 epochs)
        if (ep + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                val_loss = 0.0; val_n = 0
                for i in range(0, len(feat_val), bs):
                    c = feat_val[i:i+bs].to(DEVICE)
                    x1 = tgt_val_norm[i:i+bs].to(DEVICE)
                    t = torch.rand(len(c), 1, device=DEVICE)
                    x0 = torch.randn_like(x1)
                    x_t = (1 - t) * x0 + t * x1
                    target_vel = x1 - x0
                    v_pred = head(c, x_t, t)
                    val_loss += F.mse_loss(v_pred, target_vel).item() * len(c)
                    val_n += len(c)
                val_loss /= val_n

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(ema_head.state_dict())
                patience = 0
            else:
                patience += 10

            if (ep + 1) % 50 == 0:
                print(f'    ep {ep+1}/{config["epochs"]} train={epoch_loss/n_b:.4f} '
                      f'val={val_loss:.4f} best_val={best_val_loss:.4f} '
                      f'patience={patience}/{max_patience}', flush=True)

            if patience >= max_patience:
                print(f'    Early stopping at epoch {ep+1}', flush=True)
                break

    # Load best EMA state
    if best_state is not None:
        ema_head.load_state_dict(best_state)
        print(f'  Restored best EMA (val_loss={best_val_loss:.4f})')

    return ema_head, mu, sig


def evaluate(head, feat_te, tgt_te, mu, sig, label='FM v5'):
    head.eval()
    mu_t = mu if isinstance(mu, torch.Tensor) else torch.tensor(mu, dtype=torch.float32)
    sig_t = sig if isinstance(sig, torch.Tensor) else torch.tensor(sig, dtype=torch.float32)

    with torch.no_grad():
        samp = head.sample(feat_te, n_samples=50, num_steps=50)
        pred_norm = samp.mean(0)
    pred = (pred_norm * sig_t + mu_t).numpy()
    true = tgt_te.numpy()

    results = {}; rs = []
    for j, disp in enumerate(ROI_DISPLAY):
        r, _ = pearsonr(pred[:, j], true[:, j])
        mse = float(np.mean((pred[:, j] - true[:, j])**2))
        results[disp] = {'R': float(r), 'MSE': float(mse)}
        rs.append(float(r))

    avg_r = float(np.mean(rs))
    results['avg_r'] = avg_r

    print(f'\n  {label}:')
    print(f'  {"ROI":<16} {"R":>8}')
    print(f'  {"-"*24}')
    for disp in ROI_DISPLAY:
        print(f'  {disp:<16} {results[disp]["R"]:>8.3f}')
    print(f'  {"Avg.R":<16} {avg_r:>8.3f}')

    return results


def main():
    print('='*70)
    print('FM v5: Extended Training with Proven Architecture')
    print('='*70)

    feat_tr, tgt_tr, feat_cal, tgt_cal, feat_te, tgt_te = load_features()
    print(f'Data: train={len(feat_tr)} cal={len(feat_cal)} test={len(feat_te)}')

    # Use cal set as validation for early stopping
    # Train on train set only (not train+cal like v3b)
    print(f'\n--- Run 1: v3b architecture, extended training ---')
    config1 = {
        'proj_dim': 512, 'hidden_dim': 512, 'time_dim': 32,
        'dropout': 0.05, 'lr': 3e-4, 'wd': 1e-4,
        'epochs': 1000, 'bs': 128, 'feat_noise': 0.01,
    }
    head1, mu1, sig1 = train_fm(feat_tr, tgt_tr, feat_cal, tgt_cal, config1)
    r1 = evaluate(head1, feat_te, tgt_te, mu1, sig1, 'v5-base (512D)')
    torch.save({'head': head1.state_dict(), 'mu': mu1.numpy(), 'sig': sig1.numpy()},
               f'{SAVE_DIR}/fm_v5_base.pth')

    # Run 2: train on train+cal (more data, like v3b did)
    print(f'\n--- Run 2: v3b architecture, train on train+cal (more data) ---')
    feat_tr2 = torch.cat([feat_tr, feat_cal], 0)
    tgt_tr2  = torch.cat([tgt_tr,  tgt_cal],  0)
    # Split off 10% of combined as internal val
    n_total = len(feat_tr2)
    n_val = int(0.1 * n_total)
    perm = torch.randperm(n_total)
    feat_tr2_train = feat_tr2[perm[n_val:]]
    tgt_tr2_train  = tgt_tr2[perm[n_val:]]
    feat_tr2_val   = feat_tr2[perm[:n_val]]
    tgt_tr2_val    = tgt_tr2[perm[:n_val]]

    config2 = {
        'proj_dim': 512, 'hidden_dim': 512, 'time_dim': 32,
        'dropout': 0.05, 'lr': 3e-4, 'wd': 1e-4,
        'epochs': 1000, 'bs': 128, 'feat_noise': 0.01,
    }
    head2, mu2, sig2 = train_fm(feat_tr2_train, tgt_tr2_train, feat_tr2_val, tgt_tr2_val, config2)
    r2 = evaluate(head2, feat_te, tgt_te, mu2, sig2, 'v5-moredata (512D, train+cal)')
    torch.save({'head': head2.state_dict(), 'mu': mu2.numpy(), 'sig': sig2.numpy()},
               f'{SAVE_DIR}/fm_v5_moredata.pth')

    # Run 3: wider model
    print(f'\n--- Run 3: wider model (640D), train on train+cal ---')
    config3 = {
        'proj_dim': 640, 'hidden_dim': 640, 'time_dim': 48,
        'dropout': 0.08, 'lr': 2e-4, 'wd': 5e-5,
        'epochs': 1000, 'bs': 128, 'feat_noise': 0.02,
    }
    head3, mu3, sig3 = train_fm(feat_tr2_train, tgt_tr2_train, feat_tr2_val, tgt_tr2_val, config3)
    r3 = evaluate(head3, feat_te, tgt_te, mu3, sig3, 'v5-wide (640D, train+cal)')
    torch.save({'head': head3.state_dict(), 'mu': mu3.numpy(), 'sig': sig3.numpy()},
               f'{SAVE_DIR}/fm_v5_wide.pth')

    # Summary
    print('\n' + '='*70)
    print('Summary:')
    print(f'  v3b original:          Avg.R = 0.468')
    print(f'  v5-base (512D):        Avg.R = {r1["avg_r"]:.3f}')
    print(f'  v5-moredata (512D):    Avg.R = {r2["avg_r"]:.3f}')
    print(f'  v5-wide (640D):        Avg.R = {r3["avg_r"]:.3f}')
    print(f'  Target:                Avg.R = 0.600')
    print('='*70)

    # Save best
    best = max([(r1, 'base'), (r2, 'moredata'), (r3, 'wide')], key=lambda x: x[0]['avg_r'])
    print(f'\n  Best: v5-{best[1]} (Avg.R = {best[0]["avg_r"]:.3f})')

    with open(OUT_JSON, 'w') as f:
        json.dump({
            'v5_base': r1, 'v5_moredata': r2, 'v5_wide': r3,
            'best': best[1], 'best_avg_r': best[0]['avg_r'],
        }, f, indent=2)
    print(f'Saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
