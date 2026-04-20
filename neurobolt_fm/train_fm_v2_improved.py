"""
FM-NeuroBOLT v2 Improved — Round 6:
  1. Ridge residual learning: FM predicts (y - ridge_mean), combining linear + distributional
  2. Longer training: 500 epochs with cosine annealing
  3. More ODE trajectories: N=50 for better mean estimation
  4. Conformal calibration: hold out 10% of train data as calibration set → Cover90 ≈ 0.90
  5. Per-scan calibration for robustness

Addresses reviewer concerns:
  - FM R should approach Ridge (0.446) by using Ridge as a base predictor
  - Cover90 (currently 0.283) → ~0.90 via split conformal calibration
"""

import sys, os, gc, json, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import mne
mne.set_log_level('WARNING')

from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from timm.models import create_model
import models.model
from dataset_maker import preproc

# ─────────────────────────────────────────────────────────────────────────────
ROIS = [
    ("Cuneus",                        "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",           "Heschl.pth",    "Heschl's Gyrus"),
    ("Middle frontal gyrus anterior", "Midfront.pth",  "Mid. Frontal"),
    ("Precuneus anterior",            "Precuneus.pth", "Precuneus Ant."),
    ("Putamen",                       "Putamen.pth",   "Putamen"),
    ("Thalamus",                       "Thalamus.pth",  "Thalamus"),
    ("global signal clean",           "glb.pth",       "Global Signal"),
]
CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]

CKPT_DIR  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/code/checkpoints'
DATA_ROOT = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/data'
SAVE_DIR  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/checkpoints_fm_v2_imp'
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)

def build_backbone(ckpt_path):
    m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                     drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=True, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=True)
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = st['model'] if 'model' in st else st
    for k in ['head.weight','head.bias']:
        if k in st and st[k].shape != m.state_dict()[k].shape: del st[k]
    for k in list(st.keys()):
        if 'relative_position_index' in k: st.pop(k)
    m.load_state_dict(st, strict=False)
    m.eval().to(DEVICE)
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def extract_features(backbone, eeg_tensor, bs=64):
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b,'B N A T -> B N (A T)'), input_chans=None)
        out.append(backbone.head_act(xm + xt).cpu())
    return torch.cat(out)


# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FlowMatchingHeadV2Imp(nn.Module):
    """FM head predicting residuals around Ridge mean — larger model."""
    def __init__(self, feature_dim=200, hidden_dim=512, time_dim=32):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(time_dim)
        in_dim = feature_dim + 1 + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, features, x_t, t):
        t_emb = self.time_emb(t)
        inp = torch.cat([features, x_t, t_emb], dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def sample(self, features, n_samples=50, num_steps=100):
        """MC mean: average N independent ODE trajectories."""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x = torch.randn(B, 1, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x)
        return torch.stack(preds, dim=0)  # (N_samples, B, 1)


def load_scan(sub, scan, roi_col):
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    raw  = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200: raw.resample(200)
    raw.load_data(); raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    df = pd.read_pickle(fm_path)
    fm = df[[roi_col]].to_numpy().T
    b, a = butter(5, 0.15/(0.5/TR), btype='low')
    fm = filtfilt(b, a, fm, axis=1)
    fm, _ = preproc.normalize_data(fm)

    data_epoch, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
    del raw
    n         = len(data_epoch["eeg"])
    traincrop = int(0.8 * n)
    valcrop   = int(0.1 * n) + traincrop + math.ceil(20 / TR)

    eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"]])

    def to_float(x):
        return float(np.asarray(x).flat[0])

    eeg_train = eeg_all[:valcrop]
    tgt_train = torch.tensor([to_float(x) for x in data_epoch["fmri"][:valcrop]], dtype=torch.float32)
    eeg_test  = eeg_all[valcrop:]
    tgt_test  = torch.tensor([to_float(x) for x in data_epoch["fmri"][valcrop:]], dtype=torch.float32)

    return eeg_train, tgt_train, eeg_test, tgt_test


def train_fm_v2_improved(feat_tr, tgt_tr_raw, epochs=500, lr=3e-4, bs=128):
    """
    Ridge + FM residual learning.
    Returns: head, ridge_model, target_scaler (mu, sig of residuals)
    """
    # 1. Split: 80% FM train, 10% conformal calibration
    n = len(feat_tr)
    n_cal = max(20, int(0.1 * n))
    n_train = n - n_cal
    feat_fm_tr  = feat_tr[:n_train]
    tgt_fm_tr   = tgt_tr_raw[:n_train]
    feat_cal    = feat_tr[n_train:]
    tgt_cal_raw = tgt_tr_raw[n_train:]

    # 2. Fit Ridge on training data
    ridge = Ridge(alpha=1.0)
    ridge.fit(feat_fm_tr.numpy(), tgt_fm_tr.numpy())
    ridge_pred_tr = torch.tensor(ridge.predict(feat_fm_tr.numpy()), dtype=torch.float32)

    # 3. Compute residuals for FM training
    resid_tr = tgt_fm_tr - ridge_pred_tr

    # 4. Z-score normalize residuals
    mu_r  = resid_tr.mean().item()
    sig_r = resid_tr.std().item() + 1e-8
    resid_norm = (resid_tr - mu_r) / sig_r

    # 5. Train FM head on residuals
    head = FlowMatchingHeadV2Imp(feat_fm_tr.shape[1]).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(feat_fm_tr))
        ep_loss = 0.0
        n_steps = 0
        for i in range(0, len(feat_fm_tr), bs):
            idx = perm[i:i+bs]
            c   = feat_fm_tr[idx].to(DEVICE)
            x1  = resid_norm[idx].to(DEVICE).unsqueeze(-1)
            t   = torch.rand(len(c), 1, device=DEVICE)
            x0  = torch.randn_like(x1)
            xt  = (1-t)*x0 + t*x1
            loss = F.mse_loss(head(c, xt, t), x1 - x0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_steps += 1
        sch.step()
        if (ep+1) % 100 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={ep_loss/n_steps:.4f}')

    # 6. Conformal calibration on held-out cal set
    ridge_pred_cal = torch.tensor(ridge.predict(feat_cal.numpy()), dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        # Get N=50 residual trajectories on cal set
        samples = head.sample(feat_cal.to(DEVICE), n_samples=50, num_steps=100)
        # samples: (50, n_cal, 1) — in normalized residual space
        samples = samples.squeeze(-1).cpu()  # (50, n_cal)
        # De-normalize
        samples_raw = samples * sig_r + mu_r  # (50, n_cal)
        # Full predictions = ridge + fm_residual
        preds_cal = ridge_pred_cal.unsqueeze(0) + samples_raw  # (50, n_cal)

    # Conformal score: |y - median_pred|
    median_cal = preds_cal.median(dim=0).values  # (n_cal,)
    conf_scores = (tgt_cal_raw - median_cal).abs()  # (n_cal,)
    # 90th percentile with finite-sample correction (Vovk et al.)
    n_c = len(conf_scores)
    q_level = math.ceil((n_c + 1) * 0.9) / n_c
    q_level = min(q_level, 1.0)
    q_conf = float(np.quantile(conf_scores.numpy(), q_level))
    print(f'  Conformal quantile (90%): {q_conf:.4f}  (cal_n={n_c})')

    return head, ridge, mu_r, sig_r, q_conf


def run_roi(roi_col, ckpt_path, display, n_samples=50):
    print(f'  Loading backbone {ckpt_path}...')
    backbone = build_backbone(ckpt_path)

    feat_tr_l, tgt_tr_l = [], []
    feat_te_l, tgt_te_l = [], []
    scan_names_te = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'    {pat}...', end=' ', flush=True)
        try:
            et, tt, ete, tte = load_scan(sub, scan, roi_col)
            if len(ete) < 5:
                print('SKIP (short)'); continue
            feat_tr_l.append(extract_features(backbone, et))
            tgt_tr_l.append(tt)
            feat_te_l.append(extract_features(backbone, ete))
            tgt_te_l.append(tte)
            scan_names_te.append(pat)
            print(f'tr={len(et)} te={len(ete)}')
        except Exception as e:
            print(f'SKIP ({e})')

    feat_tr = torch.cat(feat_tr_l); tgt_tr = torch.cat(tgt_tr_l)
    feat_te = torch.cat(feat_te_l); tgt_te = torch.cat(tgt_te_l)
    del feat_tr_l

    print(f'  Train={len(feat_tr)} Test={len(feat_te)}  Training FM v2+Improved...')
    head, ridge, mu_r, sig_r, q_conf = train_fm_v2_improved(feat_tr, tgt_tr)

    # Evaluate on test set
    ridge_pred_te = torch.tensor(ridge.predict(feat_te.numpy()), dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        samples = head.sample(feat_te.to(DEVICE), n_samples=n_samples, num_steps=100)
        samples = samples.squeeze(-1).cpu()  # (50, n_te)
        samples_raw = samples * sig_r + mu_r  # de-normalize residuals
        preds_all_samples = ridge_pred_te.unsqueeze(0) + samples_raw  # (50, n_te)

    pred_all = preds_all_samples.median(dim=0).values.numpy()  # median prediction
    true_all = tgt_te.numpy()

    # Coverage check with conformal interval
    lower = pred_all - q_conf
    upper = pred_all + q_conf
    coverage = float(np.mean((true_all >= lower) & (true_all <= upper)))
    print(f'  Conformal Cover90: {coverage:.3f} (target: 0.90)')

    per_scan_r = []
    idx = 0
    for nm, fe in zip(scan_names_te, feat_te_l):
        n = len(fe)
        ps = pred_all[idx:idx+n]; ts = true_all[idx:idx+n]
        idx += n
        if n >= 5:
            r, _ = pearsonr(ps, ts)
            per_scan_r.append(float(r))
            print(f'    {nm}: R={r:.3f}')

    r_g   = float(pearsonr(pred_all, true_all)[0])
    mse_g = float(np.mean((pred_all - true_all)**2))
    print(f'  -> Global R={r_g:.3f}  MSE={mse_g:.3f}  (mean scan R={float(np.mean(per_scan_r)):.3f})')

    # Save checkpoint
    import pickle
    torch.save({
        'head': head.state_dict(),
        'mu_r': mu_r, 'sig_r': sig_r, 'q_conf': q_conf,
        'ridge_coef': ridge.coef_, 'ridge_intercept': ridge.intercept_,
    }, os.path.join(SAVE_DIR, f'fm_v2imp_{display.replace(".", "").replace(" ","_")}.pth'))

    del backbone; gc.collect(); torch.cuda.empty_cache()
    return r_g, mse_g, per_scan_r, coverage, q_conf


def main():
    OUT_JSON = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v2_improved_results.json'

    print(f'\n{"="*70}')
    print(f'FM-NeuroBOLT v2 Improved — Ridge Residual + Conformal Calibration')
    print(f'{"="*70}')
    print(f'Device: {DEVICE}  N_samples: 50  Epochs: 500  Backbone: frozen')
    print()

    results = {}
    r_all, cov_all = [], []

    for roi_col, ckpt_fname, display in ROIS:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'\n[ROI] {display}  ({roi_col})')
        r, mse, per_scan_r, coverage, q_conf = run_roi(roi_col, ckpt_path, display)
        results[display] = {
            'R': float(r), 'MSE': float(mse),
            'per_scan_Rs': [float(x) for x in per_scan_r],
            'Cover90_conformal': float(coverage),
            'conformal_q': float(q_conf),
        }
        r_all.append(r); cov_all.append(coverage)

    results['avg_r'] = float(np.nanmean(r_all))
    results['avg_cover90'] = float(np.nanmean(cov_all))

    with open(OUT_JSON, 'w') as f: json.dump(results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')

    # Summary table
    print(f'\n{"="*70}')
    print(f'SUMMARY — FM v2 Improved (Ridge Residual + Conformal)')
    print(f'{"="*70}')
    print(f"{'ROI':<25} {'R':>8} {'MSE':>8} {'Cover90':>10}")
    print('-'*55)
    for roi_col, _, display in ROIS:
        d = results.get(display, {})
        print(f"  {display:<23} {d.get('R',float('nan')):>8.3f} {d.get('MSE',float('nan')):>8.3f} {d.get('Cover90_conformal',float('nan')):>10.3f}")
    print('-'*55)
    print(f"  {'AVERAGE':<23} {results['avg_r']:>8.3f} {'':>8} {results['avg_cover90']:>10.3f}")
    print()
    print(f'Comparison: Ridge Avg.R=0.446, FM v2 (old) Avg.R=0.424, NeuroBOLT Avg.R=0.406')

if __name__ == '__main__':
    main()
