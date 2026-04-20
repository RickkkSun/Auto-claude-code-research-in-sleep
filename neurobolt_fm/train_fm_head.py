"""
Flow Matching (OT-CFM) head for NeuroBOLT — EEG-to-fMRI ROI prediction.

For each ROI:
  1. Load pretrained NeuroBOLT backbone (frozen).
  2. Iterate over 29 scans; for each scan extract backbone features
     immediately and discard raw EEG to stay memory-efficient.
  3. Train a FlowMatchingHead (small MLP velocity network).
  4. Evaluate via Euler ODE integration; report R + MSE per scan and global.
  5. Print side-by-side Table 1 vs NeuroBOLT pretrained baseline.

Usage:
    python train_fm_head.py [--data_root ./data] [--ckpt_dir ./code/checkpoints]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

import argparse, json, gc, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import mne
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from timm.models import create_model
import models.model   # registers 'neurobolt_default'
from dataset_maker import preproc

# ─────────────────────────────────────────────────────────────────────────────
# Constants
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

VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
TR    = 2.1
TMIN  = -16
CROP  = 3200
EVENT = 'R149'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_input_chans(ch_names):
    from utils import get_input_chans as _get
    return _get(ch_names)


def load_scan_features(sub_idx, scan_idx, data_root, roi_col, backbone, batch_size=64):
    """Load one scan, extract backbone features, discard raw EEG.

    Returns (feat_train, tgt_train, feat_test, tgt_test) – all CPU tensors.
    """
    patient   = f'sub{sub_idx:02d}-scan{scan_idx:02d}'
    eeg_path  = os.path.join(data_root, 'EEG', f'{patient}_eeg.set')
    fmri_path = os.path.join(data_root, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    # ── EEG ──────────────────────────────────────────────────────────────────
    raw  = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    # ── fMRI ─────────────────────────────────────────────────────────────────
    df_fmri = pd.read_pickle(fmri_path)
    fmri_np = df_fmri[[roi_col]].to_numpy().T   # (1, T_fmri)
    fs = 1 / TR; nyq = 0.5 * fs
    b, a = butter(N=5, Wn=0.15/nyq, btype='low', analog=False)
    fmri_np = filtfilt(b, a, fmri_np, axis=1)
    fmri_norm, _ = preproc.normalize_data(fmri_np)

    # ── Epoch ─────────────────────────────────────────────────────────────────
    data_epoch, _ = preproc.epoching_seq2one(
        raw, fmri_norm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
    del raw   # free MNE raw immediately

    n         = len(data_epoch["eeg"])
    traincrop = int(0.8 * n)
    valcrop   = int(0.1 * n) + traincrop + math.ceil(20 / TR)

    if traincrop < 5 or valcrop >= n:
        return None

    # 80% train, separate 10% val from end of train region
    val_split = int(0.875 * traincrop)   # ~70% train, ~10% val

    eeg_tr  = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"][:val_split]])
    tgt_tr  = torch.tensor([x.item() if hasattr(x, 'item') else float(x)
                             for x in data_epoch["fmri"][:val_split]], dtype=torch.float32)

    eeg_val = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"][val_split:traincrop]])
    tgt_val = torch.tensor([x.item() if hasattr(x, 'item') else float(x)
                             for x in data_epoch["fmri"][val_split:traincrop]], dtype=torch.float32)

    eeg_te  = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"][valcrop:]])
    tgt_te  = torch.tensor([x.item() if hasattr(x, 'item') else float(x)
                             for x in data_epoch["fmri"][valcrop:]], dtype=torch.float32)

    del data_epoch  # free epoch list

    # ── Extract backbone features (GPU → CPU) ─────────────────────────────
    feat_tr  = extract_features(backbone, eeg_tr,  batch_size)
    feat_val = extract_features(backbone, eeg_val, batch_size)
    feat_te  = extract_features(backbone, eeg_te,  batch_size)
    del eeg_tr, eeg_val, eeg_te   # free EEG tensors

    return feat_tr, tgt_tr, feat_val, tgt_val, feat_te, tgt_te


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────
def build_backbone(ckpt_path):
    model = create_model(
        'neurobolt_default',
        EEG_channel=26, num_roi=1,
        drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
        drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
        use_rel_pos_bias=True, use_abs_pos_emb=True,
        init_values=0.1, qkv_bias=True,
    )
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    for k in ['head.weight', 'head.bias']:
        if k in state and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    for key in list(state.keys()):
        if 'relative_position_index' in key:
            state.pop(key)
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(backbone, eeg_tensor, batch_size=64):
    """Forward EEG through backbone up to (not including) the linear head."""
    ic   = get_input_chans(CH_NAMES)
    feats = []
    for i in range(0, len(eeg_tensor), batch_size):
        batch = eeg_tensor[i:i+batch_size].to(DEVICE) / 100.0
        batch = rearrange(batch, 'B N (A T) -> B N A T', T=200)
        x_tmp = backbone.forward_ts_features(batch, input_chans=ic)
        x_mss = backbone.mss_module(
            rearrange(batch, 'B N A T -> B N (A T)'), input_chans=None)
        feats.append(backbone.head_act(x_mss + x_tmp).cpu())
    return torch.cat(feats)  # (N, 200)


# ─────────────────────────────────────────────────────────────────────────────
# Flow Matching Head (OT-CFM, 1-D)
# ─────────────────────────────────────────────────────────────────────────────
class FlowMatchingHead(nn.Module):
    def __init__(self, feature_dim=200, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),       nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, x_t, t):
        return self.net(torch.cat([features, x_t, t], dim=-1))

    @torch.no_grad()
    def sample(self, features, num_steps=50):
        B  = features.shape[0]
        x  = torch.randn(B, 1, device=features.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B, 1), i / num_steps, device=features.device)
            x = x + self.forward(features, x, t) * dt
        return x


def train_fm(feat_tr, tgt_tr, feat_val, tgt_val,
             hidden_dim=256, epochs=150, lr=3e-4, batch_size=64, patience=30):
    head  = FlowMatchingHead(feat_tr.shape[1], hidden_dim).to(DEVICE)
    opt   = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best_r, best_state, no_imp = -1.0, None, 0

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), batch_size):
            idx  = perm[i:i+batch_size]
            c    = feat_tr[idx].to(DEVICE)
            x_1  = tgt_tr[idx].to(DEVICE).unsqueeze(-1)
            t    = torch.rand(len(c), 1, device=DEVICE)
            x_0  = torch.randn_like(x_1)
            x_t  = (1 - t) * x_0 + t * x_1
            loss = F.mse_loss(head(c, x_t, t), x_1 - x_0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            p = head.sample(feat_val.to(DEVICE)).squeeze().cpu().numpy()
        val_r = pearsonr(p, tgt_val.numpy())[0] if len(p) >= 5 else -1.0

        if val_r > best_r:
            best_r = val_r
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if (ep + 1) % 25 == 0:
            print(f'    ep {ep+1:3d}/{epochs}  val_R={val_r:.3f}  best={best_r:.3f}')
        if no_imp >= patience:
            print(f'    Early stop at ep {ep+1}  best_val_R={best_r:.3f}')
            break

    head.load_state_dict(best_state)
    return head


# ─────────────────────────────────────────────────────────────────────────────
# Per-ROI pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_roi(roi_col, ckpt_path, data_root, args):
    print(f'  Loading backbone {ckpt_path}...')
    backbone = build_backbone(ckpt_path)

    feat_tr_list,  tgt_tr_list  = [], []
    feat_val_list, tgt_val_list = [], []
    feat_te_list,  tgt_te_list  = [], []
    scan_names_te = []

    for (sub, scan) in ALL_SCANS:
        patient = f'sub{sub:02d}-scan{scan:02d}'
        print(f'    {patient}...', end=' ', flush=True)
        try:
            result = load_scan_features(sub, scan, data_root, roi_col, backbone)
            if result is None:
                print('SKIP (too small)')
                continue
            ft, tt, fv, tv, fe, te = result
            feat_tr_list.append(ft);  tgt_tr_list.append(tt)
            feat_val_list.append(fv); tgt_val_list.append(tv)
            feat_te_list.append(fe);  tgt_te_list.append(te)
            scan_names_te.append(patient)
            print(f'tr={len(ft)} te={len(fe)}')
        except Exception as e:
            print(f'SKIP ({e})')

    if not feat_tr_list:
        return float('nan'), float('nan'), {}

    feat_tr  = torch.cat(feat_tr_list);  tgt_tr  = torch.cat(tgt_tr_list)
    feat_val = torch.cat(feat_val_list); tgt_val = torch.cat(tgt_val_list)
    feat_te  = torch.cat(feat_te_list);  tgt_te  = torch.cat(tgt_te_list)
    del feat_tr_list, feat_val_list  # save memory

    print(f'  Train={len(feat_tr)} Val={len(feat_val)} Test={len(feat_te)}  Training...')
    head = train_fm(feat_tr, tgt_tr, feat_val, tgt_val,
                    hidden_dim=args.hidden_dim, epochs=args.epochs,
                    lr=args.lr, batch_size=args.batch_size, patience=args.patience)

    # Evaluate
    head.eval()
    with torch.no_grad():
        pred_all = head.sample(feat_te.to(DEVICE), num_steps=args.num_steps)
    pred_all = pred_all.squeeze().cpu().numpy()
    true_all = tgt_te.numpy()

    per_scan_r = []
    idx = 0
    for nm, fe in zip(scan_names_te, feat_te_list):
        n  = len(fe)
        ps = pred_all[idx:idx+n]; ts = true_all[idx:idx+n]
        idx += n
        if n >= 5:
            r, _ = pearsonr(ps, ts)
            per_scan_r.append(r)
            print(f'    {nm}: R={r:.3f}  MSE={np.mean((ps-ts)**2):.3f}')

    r_g, _   = pearsonr(pred_all, true_all)
    mse_g    = float(np.mean((pred_all - true_all) ** 2))
    r_g      = float(r_g)
    print(f'  -> Global R={r_g:.3f}  MSE={mse_g:.3f}  (mean per-scan R={float(np.mean(per_scan_r)):.3f})')

    del backbone; gc.collect()
    torch.cuda.empty_cache()

    return r_g, mse_g, {'per_scan_Rs': [float(x) for x in per_scan_r]}


# ─────────────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────────────
NB_KEY = {'Cuneus':'Cuneus', "Heschl's Gyrus":'Heschl', 'Mid. Frontal':'MidFrontal',
          'Precuneus Ant.':'PrecuneusAnt', 'Putamen':'Putamen',
          'Thalamus':'Thalamus', 'Global Signal':'GlobalSig'}


def print_table(baseline_json, fm_results):
    with open(baseline_json) as f:
        nb = json.load(f)

    roi_display = [r[2] for r in ROIS]
    cw = 14
    div = '=' * (22 + cw * len(roi_display) + 10)
    print(); print(div)
    print('  NeuroBOLT vs FM-NeuroBOLT  (intra-subject, Pearson R / MSE)')
    print(div)
    hdr = f"{'Method':<22}" + ''.join(f'{d:>{cw}}' for d in roi_display) + f"{'Avg.R':>10}"
    print(hdr); print('-' * len(hdr))

    rows = {
        'NeuroBOLT (pretrained)': {d: {'R': nb[NB_KEY[d]]['R'], 'MSE': nb[NB_KEY[d]]['MSE']}
                                    for d in roi_display},
        'FM-NeuroBOLT':          {d: {'R': fm_results.get(d, {}).get('R', float('nan')),
                                       'MSE': fm_results.get(d, {}).get('MSE', float('nan'))}
                                   for d in roi_display},
    }
    for method, row in rows.items():
        r_v = [row[d]['R']   for d in roi_display]
        m_v = [row[d]['MSE'] for d in roi_display]
        avg = float(np.nanmean(r_v))
        line = f'{method:<22}'
        for r, m in zip(r_v, m_v):
            line += f'{f"{r:.3f}/{m:.3f}":>{cw}}'
        line += f'{avg:>10.3f}'
        print(line)

    print(div)
    print('\nNote: R/MSE per cell.  Avg.R = mean Pearson R across 7 ROIs.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     default='C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/data')
    parser.add_argument('--ckpt_dir',      default='C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/code/checkpoints')
    parser.add_argument('--baseline_json', default='C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/neurobolt_intra_results.json')
    parser.add_argument('--output_json',   default='C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_results.json')
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--hidden_dim', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--patience',   type=int,   default=30)
    parser.add_argument('--num_steps',  type=int,   default=50)
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'FM-NeuroBOLT: OT-CFM head on pretrained LaBraM backbone')
    print(f'{"="*60}')
    print(f'Device: {DEVICE}  Epochs: {args.epochs}  Hidden: {args.hidden_dim}  LR: {args.lr}\n')

    fm_results = {}
    r_all      = []

    for roi_col, ckpt_fname, display in ROIS:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_fname)
        print(f'\n[ROI] {display}  ({roi_col})')
        r, mse, extra = run_roi(roi_col, ckpt_path, args.data_root, args)
        fm_results[display] = {'R': r, 'MSE': mse, **extra}
        r_all.append(r)

    fm_results['avg_r'] = float(np.nanmean(r_all))

    def to_native(obj):
        """Recursively convert numpy scalars to Python native types."""
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(args.output_json, 'w') as f:
        json.dump(to_native(fm_results), f, indent=2)
    print(f'\nSaved → {args.output_json}')

    print_table(args.baseline_json, fm_results)


if __name__ == '__main__':
    main()
