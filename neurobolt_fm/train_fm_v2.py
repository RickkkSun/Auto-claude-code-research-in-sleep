"""
FM-NeuroBOLT v2 — Round 1 improvements:
  1. Train on ALL data before the test split (90%+ per scan vs 70% in v1)
  2. Larger FM head (512 hidden, 4 layers)
  3. Target z-score standardisation across all training samples
  4. 200 epochs, no early stopping (ignore overfitting per user instruction)
  5. Multi-sample inference (N=20 ODE trajectories averaged)
  6. Save per-ROI model checkpoints

Evaluation is identical to evaluate_neurobolt.py (same 10% test split).
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
SAVE_DIR  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/checkpoints_fm_v2'
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
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
# FM Head v2 — larger model, SiLU activations, time sinusoidal embedding
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B, 1) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class FlowMatchingHeadV2(nn.Module):
    """Larger FM head with sinusoidal time embedding and SiLU activations."""
    def __init__(self, feature_dim=200, hidden_dim=512, time_dim=32):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(time_dim)
        in_dim = feature_dim + 1 + time_dim  # features + x_t + t_emb
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
    def sample(self, features, n_samples=20, num_steps=100):
        """Monte-Carlo mean: average N independent ODE trajectories."""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x = torch.randn(B, 1, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x)
        return torch.stack(preds).mean(0)  # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as evaluate_neurobolt.py — consistent test split)
# ─────────────────────────────────────────────────────────────────────────────
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

    eeg_all   = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"]])

    # v2 key change: train on EVERYTHING before the test window (≈90% of scan)
    def to_float(x):
        import numpy as np
        return float(np.asarray(x).flat[0])

    eeg_train = eeg_all[:valcrop]
    tgt_train = torch.tensor([to_float(x) for x in data_epoch["fmri"][:valcrop]], dtype=torch.float32)
    eeg_test  = eeg_all[valcrop:]
    tgt_test  = torch.tensor([to_float(x) for x in data_epoch["fmri"][valcrop:]], dtype=torch.float32)

    return eeg_train, tgt_train, eeg_test, tgt_test


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_fm_v2(feat_tr, tgt_tr_raw, hidden_dim=512, epochs=200, lr=3e-4, bs=128):
    """OT-CFM training with z-score target normalisation."""
    # z-score normalise targets
    mu  = tgt_tr_raw.mean().item()
    sig = tgt_tr_raw.std().item() + 1e-8
    tgt_tr = (tgt_tr_raw - mu) / sig

    head = FlowMatchingHeadV2(feat_tr.shape[1], hidden_dim).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(feat_tr) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                              total_steps=epochs * steps_per_epoch)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            c   = feat_tr[idx].to(DEVICE)
            x1  = tgt_tr[idx].to(DEVICE).unsqueeze(-1)
            t   = torch.rand(len(c), 1, device=DEVICE)
            x0  = torch.randn_like(x1)
            xt  = (1-t)*x0 + t*x1
            loss = F.mse_loss(head(c, xt, t), x1 - x0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sch.step()

        if (ep+1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={loss.item():.4f}')

    return head, mu, sig


# ─────────────────────────────────────────────────────────────────────────────
# Per-ROI pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_roi(roi_col, ckpt_path, display, n_samples=20):
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
                print('SKIP (short)')
                continue
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

    print(f'  Train={len(feat_tr)} Test={len(feat_te)}  Training FM v2...')
    head, mu, sig = train_fm_v2(feat_tr, tgt_tr)

    # Evaluate with multi-sample inference
    head.eval()
    with torch.no_grad():
        pred_norm = head.sample(feat_te.to(DEVICE), n_samples=n_samples, num_steps=100)
    pred_norm = pred_norm.squeeze().cpu().numpy()
    pred_all  = pred_norm * sig + mu   # de-normalise
    true_all  = tgt_te.numpy()

    per_scan_r = []
    idx = 0
    for nm, fe in zip(scan_names_te, feat_te_l):
        n  = len(fe)
        ps = pred_all[idx:idx+n]; ts = true_all[idx:idx+n]
        idx += n
        if n >= 5:
            r, _ = pearsonr(ps, ts)
            per_scan_r.append(float(r))
            print(f'    {nm}: R={r:.3f}  MSE={float(np.mean((ps-ts)**2)):.3f}')

    r_g  = float(pearsonr(pred_all, true_all)[0])
    mse_g = float(np.mean((pred_all - true_all)**2))
    print(f'  -> Global R={r_g:.3f}  MSE={mse_g:.3f}  (mean scan R={float(np.mean(per_scan_r)):.3f})')

    # Save checkpoint
    torch.save({'head': head.state_dict(), 'mu': mu, 'sig': sig},
               os.path.join(SAVE_DIR, f'fm_v2_{display.replace(".", "").replace(" ","_")}.pth'))

    del backbone; gc.collect(); torch.cuda.empty_cache()
    return r_g, mse_g, per_scan_r


# ─────────────────────────────────────────────────────────────────────────────
# Table printing (paper format)
# ─────────────────────────────────────────────────────────────────────────────
NB_KEY = {'Cuneus':'Cuneus', "Heschl's Gyrus":'Heschl', 'Mid. Frontal':'MidFrontal',
          'Precuneus Ant.':'PrecuneusAnt', 'Putamen':'Putamen',
          'Thalamus':'Thalamus', 'Global Signal':'GlobalSig'}

def print_table(nb_json, v1_json, v2_results):
    with open(nb_json) as f: nb = json.load(f)
    with open(v1_json) as f: v1 = json.load(f)

    roi_display = [r[2] for r in ROIS]
    cw = 15
    div = '=' * (28 + cw * len(roi_display) + 10)
    print(); print(div)
    print('  NeuroBOLT Paper Table 1 Reproduction + Flow Matching Comparison (Pearson R / MSE)')
    print(div)
    hdr = f"{'Method':<28}" + ''.join(f'{d:>{cw}}' for d in roi_display) + f"{'Avg.R':>10}"
    print(hdr); print('-' * len(hdr))

    rows = {
        'NeuroBOLT (pretrained)': {
            d: {'R': nb[NB_KEY[d]]['R'], 'MSE': nb[NB_KEY[d]]['MSE']} for d in roi_display},
        'FM v1 (frozen, 70% data)': {
            d: {'R': v1.get(d, {}).get('R', float('nan')),
                'MSE': v1.get(d, {}).get('MSE', float('nan'))} for d in roi_display},
        'FM v2 (frozen, all data, N=20)': {
            d: {'R': v2_results.get(d, {}).get('R', float('nan')),
                'MSE': v2_results.get(d, {}).get('MSE', float('nan'))} for d in roi_display},
    }

    for method, row in rows.items():
        rv = [row[d]['R']   for d in roi_display]
        mv = [row[d]['MSE'] for d in roi_display]
        avg = float(np.nanmean(rv))
        line = f'{method:<28}'
        for r, m in zip(rv, mv): line += f'{f"{r:.3f}/{m:.3f}":>{cw}}'
        line += f'{avg:>10.3f}'
        print(line)

    print(div)
    print('\nNote: R/MSE per cell.  Avg.R = mean Pearson R across 7 ROIs.')
    print('FM v2 changes vs v1: train on all data pre-test (~90%),')
    print('  larger head (512-256-128-64), sinusoidal time embed, N=20 MC samples.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    NB_JSON = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/neurobolt_intra_results.json'
    V1_JSON = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_results.json'
    OUT_JSON = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v2_results.json'

    print(f'\n{"="*60}')
    print(f'FM-NeuroBOLT v2 — Round 1 improvements')
    print(f'{"="*60}')
    print(f'Device: {DEVICE}  N_samples: 20  Epochs: 200')
    print()

    v2_results = {}
    r_all = []

    for roi_col, ckpt_fname, display in ROIS:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'\n[ROI] {display}  ({roi_col})')
        r, mse, per_scan_r = run_roi(roi_col, ckpt_path, display)
        v2_results[display] = {'R': float(r), 'MSE': float(mse),
                               'per_scan_Rs': [float(x) for x in per_scan_r]}
        r_all.append(r)

    v2_results['avg_r'] = float(np.nanmean(r_all))

    with open(OUT_JSON, 'w') as f: json.dump(v2_results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')

    print_table(NB_JSON, V1_JSON, v2_results)

if __name__ == '__main__':
    main()
