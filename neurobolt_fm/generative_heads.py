"""
generative_heads.py — 7 conditional generative heads for ablation study.

All heads share the same interface:
  train_xxx(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim, epochs, device)
  → returns HeadWrapper with .sample(features, n_samples, bs) → [K, N, 7] normalized

Heads implemented:
  1. DDPM        — reuse from train_comparison_ddpm.py
  2. EDM         — Karras et al. 2022 preconditioning + Heun sampler
  3. Score SDE   — VP-SDE continuous score matching + reverse SDE
  4. Consistency — Song et al. 2023 consistency training (no teacher)
  5. CVAE        — Conditional VAE (ELBO)
  6. cGAN        — WGAN-GP conditional GAN
  7. Rectified Flow — CFM + 1 reflow iteration
  8. CFM         — Conditional Flow Matching (linear interpolation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy

N_ROIS = 7

# ─── Shared utilities ─────────────────────────────────────────────────────────

class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        emb = t.float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def _crps_fast(samples, targets):
    """Fast approximate CRPS for validation."""
    K = samples.shape[0]
    e1 = (samples - targets.unsqueeze(0)).abs().mean()
    perm = torch.randperm(K)
    e2 = 0.5 * (samples - samples[perm]).abs().mean()
    return (e1 - e2).item()


def _make_feat_proj(feat_dim, proj_dim=256):
    return nn.Sequential(
        nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
        nn.Linear(proj_dim, proj_dim), nn.LayerNorm(proj_dim), nn.SiLU(),
    )


class HeadWrapper:
    """Standard wrapper for evaluation compatibility."""
    def __init__(self, sample_fn, mu_tgt, sig_tgt):
        self._sample_fn = sample_fn
        self.mu_tgt = mu_tgt
        self.sig_tgt = sig_tgt

    def sample(self, features, n_samples=50, bs=256):
        """Returns [K, N, 7] NORMALIZED samples on CPU."""
        return self._sample_fn(features, n_samples, bs)


def _normalize_targets(fmri_tr, fmri_val=None):
    mu = fmri_tr.mean(0)
    sig = fmri_tr.std(0).clamp(min=1e-8)
    tr_n = (fmri_tr - mu) / sig
    if fmri_val is not None:
        val_n = (fmri_val - mu) / sig
        return tr_n, val_n, mu, sig
    return tr_n, mu, sig


def _train_loop(model, feat_tr, fmri_tr_n, feat_val, fmri_val, mu_tgt, sig_tgt,
                loss_fn, sample_fn_factory, epochs, lr, bs, device,
                patience=30, val_interval=10, extra_models=None):
    """
    Generic training loop with CRPS-based early stopping.
    loss_fn(model, feat_batch, fmri_batch, device, extra_models) → loss tensor
    sample_fn_factory(model) → fn(features, n_samples, bs) → [K, N, 7] normalized CPU
    extra_models: dict of additional models (e.g., critic for GAN) — NOT used for param counting
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_crps, best_state, no_imp = 1e9, None, 0
    N = len(feat_tr)

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(N)
        for j in range(0, N, bs):
            b = idx[j:j+bs]
            loss = loss_fn(model, feat_tr[b].to(device), fmri_tr_n[b].to(device),
                           device, extra_models)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                samps = sample_fn_factory(model)(feat_val.to(device), 10, bs)
            val_samps = samps * sig_tgt + mu_tgt
            val_crps = _crps_fast(val_samps, fmri_val)
            if val_crps < best_crps:
                best_crps = val_crps
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDM — Karras et al. 2022
# ═══════════════════════════════════════════════════════════════════════════════

class EDMDenoiser(nn.Module):
    sigma_data = 0.5

    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512, n_rois=N_ROIS):
        super().__init__()
        self.n_rois = n_rois
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)
        in_dim = proj_dim + n_rois + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, x_noisy, sigma, features):
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out  = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in   = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.log()

        f_proj = self.feat_proj(features)
        inp = torch.cat([f_proj, c_in * x_noisy, c_noise], dim=-1)
        F_x = self.net(inp)
        return c_skip * x_noisy + c_out * F_x


@torch.no_grad()
def _edm_sample(model, features, n_samples, bs,
                sigma_min=0.002, sigma_max=80.0, num_steps=50, rho=7):
    N = features.shape[0]
    device = features.device
    # Karras time schedule
    inv_rho = 1.0 / rho
    steps = torch.arange(num_steps + 1, device=device, dtype=torch.float64)
    t_steps = (sigma_max**inv_rho + steps / num_steps *
               (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
    t_steps = t_steps.float()

    all_samples = []
    for _ in range(n_samples):
        parts = []
        for i in range(0, N, bs):
            fb = features[i:i+bs]
            B = len(fb)
            x = torch.randn(B, model.n_rois, device=device) * t_steps[0]

            for k in range(num_steps):
                t_cur, t_next = t_steps[k], t_steps[k + 1]
                sig = torch.full((B,), t_cur.item(), device=device)
                d_cur = (x - model(x, sig, fb)) / t_cur
                x_next = x + (t_next - t_cur) * d_cur
                # Heun correction
                if t_next > 0:
                    sig_n = torch.full((B,), t_next.item(), device=device)
                    d_next = (x_next - model(x_next, sig_n, fb)) / t_next
                    x_next = x + (t_next - t_cur) * 0.5 * (d_cur + d_next)
                x = x_next
            parts.append(x.cpu())
        all_samples.append(torch.cat(parts))
    return torch.stack(all_samples)


def train_edm(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
              epochs=300, lr=3e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu, sig = _normalize_targets(fmri_tr, fmri_val)
    model = EDMDenoiser(feat_dim=feat_dim).to(device)

    P_mean, P_std = -1.2, 1.2
    sigma_data = model.sigma_data

    def loss_fn(m, fb, yb, dev, _):
        log_sigma = torch.randn(len(fb), 1, device=dev) * P_std + P_mean
        sigma = log_sigma.exp().clamp(0.002, 80.0)
        noise = torch.randn_like(yb)
        x_noisy = yb + sigma * noise
        D_x = m(x_noisy, sigma.squeeze(-1), fb)
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        return (weight * (D_x - yb)**2).mean()

    def sample_factory(m):
        def fn(feats, ns, b):
            m.eval()
            return _edm_sample(m, feats, ns, b)
        return fn

    model = _train_loop(model, feat_tr, fmri_tr_n, feat_val, fmri_val,
                        mu, sig, loss_fn, sample_factory, epochs, lr, bs, device)

    def final_sample(feats, ns, b):
        model.eval()
        with torch.no_grad():
            return _edm_sample(model, feats.to(device), ns, b)
    return HeadWrapper(final_sample, mu, sig)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Score SDE — VP-SDE (Song et al. 2021)
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreNet(nn.Module):
    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512, n_rois=N_ROIS, time_dim=64):
        super().__init__()
        self.n_rois = n_rois
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)
        self.time_emb = nn.Sequential(
            SinusoidalEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        in_dim = proj_dim + n_rois + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, x_t, t, features):
        f = self.feat_proj(features)
        te = self.time_emb(t)
        return self.net(torch.cat([f, x_t, te], dim=-1))


class VPSDE:
    """VP-SDE: β(t) = β_min + t(β_max - β_min)"""
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def marginal_prob(self, t):
        """Returns (mean_coeff, std) for p(x_t | x_0)."""
        log_mean = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean))
        return mean_coeff, std

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)


@torch.no_grad()
def _sde_sample(score_net, sde, features, n_samples, bs, num_steps=200, eps=1e-3):
    """Reverse SDE sampling (Euler-Maruyama)."""
    N = features.shape[0]
    device = features.device
    dt = 1.0 / num_steps
    all_samples = []
    for _ in range(n_samples):
        parts = []
        for i in range(0, N, bs):
            fb = features[i:i+bs]
            B = len(fb)
            x = torch.randn(B, score_net.n_rois, device=device)
            for step in range(num_steps):
                t_val = 1.0 - step * dt
                t = torch.full((B, 1), t_val, device=device)
                beta_t = sde.beta(t)
                score = score_net(x, t, fb)
                # Reverse SDE: dx = [-0.5*β*x - β*score]*dt + √β*dw
                drift = -0.5 * beta_t * x - beta_t * score
                diffusion = torch.sqrt(beta_t)
                noise = torch.randn_like(x) if step < num_steps - 1 else 0
                x = x + drift * (-dt) + diffusion * math.sqrt(dt) * noise
            parts.append(x.cpu())
        all_samples.append(torch.cat(parts))
    return torch.stack(all_samples)


def train_score_sde(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
                    epochs=300, lr=3e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu, sig = _normalize_targets(fmri_tr, fmri_val)
    model = ScoreNet(feat_dim=feat_dim).to(device)
    sde = VPSDE()
    eps = 1e-5

    def loss_fn(m, fb, yb, dev, _):
        t = torch.rand(len(fb), 1, device=dev) * (1.0 - eps) + eps
        mean_coeff, std = sde.marginal_prob(t)
        noise = torch.randn_like(yb)
        x_t = mean_coeff * yb + std * noise
        score_pred = m(x_t, t, fb)
        target = -noise / std
        # Weighted score matching: weight = std^2
        loss = ((score_pred - target)**2 * std**2).mean()
        return loss

    def sample_factory(m):
        def fn(feats, ns, b):
            m.eval()
            return _sde_sample(m, sde, feats, ns, b)
        return fn

    model = _train_loop(model, feat_tr, fmri_tr_n, feat_val, fmri_val,
                        mu, sig, loss_fn, sample_factory, epochs, lr, bs, device)

    def final_sample(feats, ns, b):
        model.eval()
        with torch.no_grad():
            return _sde_sample(model, sde, feats.to(device), ns, b)
    return HeadWrapper(final_sample, mu, sig)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Consistency Model — Song et al. 2023 (Consistency Training)
# ═══════════════════════════════════════════════════════════════════════════════

class ConsistencyNet(nn.Module):
    sigma_data = 0.5

    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512, n_rois=N_ROIS):
        super().__init__()
        self.n_rois = n_rois
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)
        in_dim = proj_dim + n_rois + 1  # features + x + sigma_emb
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, x, sigma, features):
        """Consistency function: f(x, σ, c) → x_0 estimate."""
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out  = self.sigma_data * sigma / (sigma**2 + self.sigma_data**2).sqrt()
        c_in   = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.log()

        f = self.feat_proj(features)
        inp = torch.cat([f, c_in * x, c_noise], dim=-1)
        F_x = self.net(inp)
        return c_skip * x + c_out * F_x


def train_consistency(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
                      epochs=300, lr=3e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu, sig = _normalize_targets(fmri_tr, fmri_val)
    model = ConsistencyNet(feat_dim=feat_dim).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_crps, best_state, no_imp = 1e9, None, 0
    N_data = len(feat_tr)

    sigma_min, sigma_max = 0.002, 80.0
    N_steps_start, N_steps_end = 2, 150
    ema_rate = 0.999

    for ep in range(epochs):
        model.train()
        # Adaptive schedule: increase N_steps during training
        N_steps = int(N_steps_start + (N_steps_end - N_steps_start) *
                      min(ep / (epochs * 0.8), 1.0))
        # Discrete sigma schedule
        rho = 7.0
        sigmas = [(sigma_max ** (1/rho) + i / N_steps *
                   (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
                  for i in range(N_steps + 1)]

        idx = torch.randperm(N_data)
        for j in range(0, N_data, bs):
            b = idx[j:j+bs]
            fb = feat_tr[b].to(device)
            x_0 = fmri_tr_n[b].to(device)
            B = len(fb)

            # Sample adjacent timestep pair
            n_idx = torch.randint(0, N_steps, (B,))
            sigma_cur  = torch.tensor([sigmas[n.item()] for n in n_idx],
                                      device=device)
            sigma_next = torch.tensor([sigmas[n.item() + 1] for n in n_idx],
                                       device=device)

            noise = torch.randn_like(x_0)
            x_cur  = x_0 + sigma_cur.unsqueeze(-1)  * noise
            x_next = x_0 + sigma_next.unsqueeze(-1) * noise  # same noise!

            # Consistency loss
            pred_cur  = model(x_cur, sigma_cur, fb)
            with torch.no_grad():
                pred_next = ema_model(x_next, sigma_next, fb)

            loss = F.mse_loss(pred_cur, pred_next.detach())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # EMA update
            with torch.no_grad():
                for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                    p_ema.data.mul_(ema_rate).add_(p.data, alpha=1 - ema_rate)

        sched.step()

        # Validation
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samps = _consistency_sample(model, feat_val.to(device), 10, bs,
                                            sigma_max=sigma_max)
            val_samps = samps * sig + mu
            val_crps = _crps_fast(val_samps, fmri_val)
            if val_crps < best_crps:
                best_crps = val_crps
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 30:
                break

    if best_state:
        model.load_state_dict(best_state)

    def final_sample(feats, ns, b):
        model.eval()
        with torch.no_grad():
            return _consistency_sample(model, feats.to(device), ns, b,
                                       sigma_max=sigma_max)
    return HeadWrapper(final_sample, mu, sig)


@torch.no_grad()
def _consistency_sample(model, features, n_samples, bs, sigma_max=80.0):
    """Single-step consistency sampling: x_T → f(x_T, T) = x_0."""
    N = features.shape[0]
    device = features.device
    all_samples = []
    for _ in range(n_samples):
        parts = []
        for i in range(0, N, bs):
            fb = features[i:i+bs]
            B = len(fb)
            x_T = torch.randn(B, model.n_rois, device=device) * sigma_max
            sigma = torch.full((B,), sigma_max, device=device)
            x_0 = model(x_T, sigma, fb)
            parts.append(x_0.cpu())
        all_samples.append(torch.cat(parts))
    return torch.stack(all_samples)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CVAE — Conditional Variational Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalVAE(nn.Module):
    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512,
                 latent_dim=32, n_rois=N_ROIS):
        super().__init__()
        self.n_rois = n_rois
        self.latent_dim = latent_dim
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)

        # Encoder: [feat_proj, fmri] → μ, logσ²
        self.encoder = nn.Sequential(
            nn.Linear(proj_dim + n_rois, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
        )
        self.enc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: [feat_proj, z] → fmri
        self.decoder = nn.Sequential(
            nn.Linear(proj_dim + latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def encode(self, feat_proj, fmri):
        h = self.encoder(torch.cat([feat_proj, fmri], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, feat_proj, z):
        return self.decoder(torch.cat([feat_proj, z], dim=-1))

    def forward(self, features, fmri):
        fp = self.feat_proj(features)
        mu, logvar = self.encode(fp, fmri)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        recon = self.decode(fp, z)
        return recon, mu, logvar


def train_cvae(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
               epochs=300, lr=3e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu_tgt, sig_tgt = _normalize_targets(fmri_tr, fmri_val)
    model = ConditionalVAE(feat_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_crps, best_state, no_imp = 1e9, None, 0
    N = len(feat_tr)

    # KL annealing: β linearly from 0 to 1 over first 30% of training
    beta_max = 1.0

    for ep in range(epochs):
        model.train()
        beta = min(1.0, ep / (0.3 * epochs)) * beta_max
        idx = torch.randperm(N)
        for j in range(0, N, bs):
            b = idx[j:j+bs]
            fb = feat_tr[b].to(device)
            yb = fmri_tr_n[b].to(device)

            recon, z_mu, z_logvar = model(fb, yb)
            recon_loss = F.mse_loss(recon, yb)
            kl_loss = -0.5 * (1 + z_logvar - z_mu**2 - z_logvar.exp()).mean()
            loss = recon_loss + beta * kl_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samps = _cvae_sample(model, feat_val.to(device), 10, bs)
            val_samps = samps * sig_tgt + mu_tgt
            val_crps = _crps_fast(val_samps, fmri_val)
            if val_crps < best_crps:
                best_crps = val_crps
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 30:
                break

    if best_state:
        model.load_state_dict(best_state)

    def final_sample(feats, ns, b):
        model.eval()
        with torch.no_grad():
            return _cvae_sample(model, feats.to(device), ns, b)
    return HeadWrapper(final_sample, mu_tgt, sig_tgt)


@torch.no_grad()
def _cvae_sample(model, features, n_samples, bs):
    N = features.shape[0]
    device = features.device
    all_samples = []
    for _ in range(n_samples):
        parts = []
        for i in range(0, N, bs):
            fb = features[i:i+bs]
            B = len(fb)
            fp = model.feat_proj(fb)
            z = torch.randn(B, model.latent_dim, device=device)
            x = model.decode(fp, z)
            parts.append(x.cpu())
        all_samples.append(torch.cat(parts))
    return torch.stack(all_samples)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. cGAN — WGAN-GP Conditional GAN
# ═══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512,
                 noise_dim=32, n_rois=N_ROIS):
        super().__init__()
        self.n_rois = n_rois
        self.noise_dim = noise_dim
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(proj_dim + noise_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, features, z):
        fp = self.feat_proj(features)
        return self.net(torch.cat([fp, z], dim=-1))


class Critic(nn.Module):
    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512, n_rois=N_ROIS):
        super().__init__()
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(proj_dim + n_rois, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, fmri):
        fp = self.feat_proj(features)
        return self.net(torch.cat([fp, fmri], dim=-1))


def _gradient_penalty(critic, features, real, fake, device):
    alpha = torch.rand(len(real), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(features, interp)
    grads = torch.autograd.grad(d_interp, interp,
                                 grad_outputs=torch.ones_like(d_interp),
                                 create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_cgan(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
               epochs=300, lr=1e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu_tgt, sig_tgt = _normalize_targets(fmri_tr, fmri_val)
    gen = Generator(feat_dim=feat_dim).to(device)
    crit = Critic(feat_dim=feat_dim).to(device)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-4)
    opt_c = torch.optim.AdamW(crit.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=epochs)

    best_crps, best_state, no_imp = 1e9, None, 0
    N = len(feat_tr)
    n_critic = 5
    gp_lambda = 10.0

    for ep in range(epochs):
        gen.train(); crit.train()
        idx = torch.randperm(N)
        for j in range(0, N, bs):
            b = idx[j:j+bs]
            fb = feat_tr[b].to(device)
            yb = fmri_tr_n[b].to(device)
            B = len(fb)

            # Critic step
            for _ in range(n_critic):
                z = torch.randn(B, gen.noise_dim, device=device)
                fake = gen(fb, z).detach()
                c_real = crit(fb, yb).mean()
                c_fake = crit(fb, fake).mean()
                gp = _gradient_penalty(crit, fb, yb, fake, device)
                loss_c = c_fake - c_real + gp_lambda * gp
                opt_c.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(crit.parameters(), 1.0)
                opt_c.step()

            # Generator step
            z = torch.randn(B, gen.noise_dim, device=device)
            fake = gen(fb, z)
            loss_g = -crit(fb, fake).mean()
            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_g.step()

        sched_g.step(); sched_c.step()

        if (ep + 1) % 10 == 0:
            gen.eval()
            with torch.no_grad():
                samps = _cgan_sample(gen, feat_val.to(device), 10, bs)
            val_samps = samps * sig_tgt + mu_tgt
            val_crps = _crps_fast(val_samps, fmri_val)
            if val_crps < best_crps:
                best_crps = val_crps
                best_state = {k: v.cpu().clone() for k, v in gen.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 30:
                break

    if best_state:
        gen.load_state_dict(best_state)

    def final_sample(feats, ns, b):
        gen.eval()
        with torch.no_grad():
            return _cgan_sample(gen, feats.to(device), ns, b)
    return HeadWrapper(final_sample, mu_tgt, sig_tgt)


@torch.no_grad()
def _cgan_sample(gen, features, n_samples, bs):
    N = features.shape[0]
    device = features.device
    all_samples = []
    for _ in range(n_samples):
        parts = []
        for i in range(0, N, bs):
            fb = features[i:i+bs]
            B = len(fb)
            z = torch.randn(B, gen.noise_dim, device=device)
            x = gen(fb, z)
            parts.append(x.cpu())
        all_samples.append(torch.cat(parts))
    return torch.stack(all_samples)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Rectified Flow — CFM + 1 reflow iteration (Liu et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

class FlowVelocityNet(nn.Module):
    """Shared velocity network for CFM and Rectified Flow."""
    def __init__(self, feat_dim=200, proj_dim=256, hidden_dim=512, n_rois=N_ROIS, time_dim=32):
        super().__init__()
        self.n_rois = n_rois
        self.feat_proj = _make_feat_proj(feat_dim, proj_dim)
        self.time_emb = nn.Sequential(
            SinusoidalEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.roi_interact = nn.Linear(n_rois, n_rois)
        in_dim = proj_dim + n_rois + time_dim
        self.velocity_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_rois),
        )

    def forward(self, features, x_t, t):
        f = self.feat_proj(features)
        te = self.time_emb(t)
        x_i = self.roi_interact(x_t)
        return self.velocity_net(torch.cat([f, x_i, te], dim=-1))


@torch.no_grad()
def _flow_sample(model, features, n_samples, bs, num_steps=100):
    """Euler ODE integration for flow-based models."""
    N = features.shape[0]
    device = features.device
    dt = 1.0 / num_steps
    all_samples = []
    for _ in range(n_samples):
        parts = []
        for i in range(0, N, bs):
            fb = features[i:i+bs]
            B = len(fb)
            x = torch.randn(B, model.n_rois, device=device)
            for step in range(num_steps):
                t = torch.full((B, 1), step / num_steps, device=device)
                x = x + model(fb, x, t) * dt
            parts.append(x.cpu())
        all_samples.append(torch.cat(parts))
    return torch.stack(all_samples)


def _train_flow_model(model, feat_tr, fmri_tr_n, feat_val, fmri_val,
                      mu, sig, epochs, lr, bs, device):
    """Train a flow velocity model with linear interpolation."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_crps, best_state, no_imp = 1e9, None, 0
    N = len(feat_tr)

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(N)
        for j in range(0, N, bs):
            b = idx[j:j+bs]
            fb = feat_tr[b].to(device)
            x1 = fmri_tr_n[b].to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(fb), 1, device=device)
            x_t = (1 - t) * x0 + t * x1
            target = x1 - x0
            pred = model(fb, x_t, t)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samps = _flow_sample(model, feat_val.to(device), 10, bs)
            val_samps = samps * sig + mu
            val_crps = _crps_fast(val_samps, fmri_val)
            if val_crps < best_crps:
                best_crps = val_crps
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 30:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def train_rectified_flow(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
                         epochs=300, lr=3e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu, sig = _normalize_targets(fmri_tr, fmri_val)

    # Phase 1: standard flow matching
    model_v1 = FlowVelocityNet(feat_dim=feat_dim).to(device)
    model_v1 = _train_flow_model(model_v1, feat_tr, fmri_tr_n, feat_val, fmri_val,
                                  mu, sig, epochs // 2, lr, bs, device)

    # Phase 2: reflow — generate synthetic targets from v1, retrain
    model_v1.eval()
    with torch.no_grad():
        # For each training sample, generate x1' via ODE from noise
        reflow_targets = []
        for i in range(0, len(feat_tr), bs):
            fb = feat_tr[i:i+bs].to(device)
            B = len(fb)
            x = torch.randn(B, N_ROIS, device=device)
            dt = 1.0 / 100
            for step in range(100):
                t = torch.full((B, 1), step / 100, device=device)
                x = x + model_v1(fb, x, t) * dt
            reflow_targets.append(x.cpu())
        reflow_targets = torch.cat(reflow_targets)

    # Phase 2: train new model on (noise → reflow_targets) pairs
    model_v2 = FlowVelocityNet(feat_dim=feat_dim).to(device)
    model_v2 = _train_flow_model(model_v2, feat_tr, reflow_targets, feat_val, fmri_val,
                                  mu, sig, epochs // 2, lr, bs, device)

    def final_sample(feats, ns, b):
        model_v2.eval()
        with torch.no_grad():
            return _flow_sample(model_v2, feats.to(device), ns, b)
    return HeadWrapper(final_sample, mu, sig)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CFM — Conditional Flow Matching (our method, single backbone version)
# ═══════════════════════════════════════════════════════════════════════════════

def train_cfm(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
              epochs=300, lr=3e-4, bs=256, device='cuda'):
    fmri_tr_n, fmri_val_n, mu, sig = _normalize_targets(fmri_tr, fmri_val)
    model = FlowVelocityNet(feat_dim=feat_dim).to(device)
    model = _train_flow_model(model, feat_tr, fmri_tr_n, feat_val, fmri_val,
                               mu, sig, epochs, lr, bs, device)

    def final_sample(feats, ns, b):
        model.eval()
        with torch.no_grad():
            return _flow_sample(model, feats.to(device), ns, b)
    return HeadWrapper(final_sample, mu, sig)


# ─── Registry ─────────────────────────────────────────────────────────────────

HEAD_REGISTRY = {
    # 'ddpm' is handled separately via train_comparison_ddpm.train_ddpm
    'edm':              train_edm,
    'score_sde':        train_score_sde,
    'consistency':      train_consistency,
    'cvae':             train_cvae,
    'cgan':             train_cgan,
    'rectified_flow':   train_rectified_flow,
    'cfm':              train_cfm,
}

HEAD_DISPLAY = {
    'ddpm':             'DDPM',
    'edm':              'EDM',
    'score_sde':        'Score SDE',
    'consistency':      'Consistency',
    'cvae':             'CVAE',
    'cgan':             'cGAN',
    'rectified_flow':   'Rectified Flow',
    'cfm':              'CFM (Ours)',
}

ALL_HEADS = list(HEAD_DISPLAY.keys())
