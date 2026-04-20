"""
Evaluate pretrained NeuroBOLT checkpoints (intra-subject mode).
Reproduces Table 1 (upper half: intra-subject) from the paper.

Usage:
    python evaluate_neurobolt.py --data_root ./data --ckpt_dir ./code/checkpoints
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

import argparse
import math
import re
import warnings
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from functools import partial
import mne
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

from timm.models import create_model
import models.model  # registers 'neurobolt_default'
from dataset_maker import preproc

# ──────────────────────────────────────────────────────────────────────────────
# ROI definitions: (DataFrame column name, checkpoint filename, display name)
# ──────────────────────────────────────────────────────────────────────────────
ROIS = [
    ("Cuneus",                       "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",          "Heschl.pth",    "Heschl's Gyrus"),
    ("Middle frontal gyrus anterior","Midfront.pth",  "Mid. Frontal"),
    ("Precuneus anterior",           "Precuneus.pth", "Precuneus Ant."),
    ("Putamen",                      "Putamen.pth",   "Putamen"),
    ("Thalamus",                     "Thalamus.pth",  "Thalamus"),
    ("global signal clean",          "glb.pth",       "Global Signal"),
]

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']

VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
TR    = 2.1
TMIN  = -16
CROP  = 3200    # 16s * 200Hz
EVENT = 'R149'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_input_chans(ch_names):
    """Map channel names to LaBraM pos_embed indices.
    Returns [0] for cls token + 1-based indices into standard_1020 for each channel.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
    from utils import get_input_chans as _get
    return _get(ch_names)


def load_scan(sub_idx, scan_idx, data_root, roi_col):
    """Load one scan: return (eeg_test_tensor, fmri_test_tensor)."""
    patient = f'sub{sub_idx:02d}-scan{scan_idx:02d}'
    eeg_path  = os.path.join(data_root, 'EEG', f'{patient}_eeg.set')
    fmri_path = os.path.join(data_root, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    # ── EEG ──
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False)
    # Drop non-EEG channels depending on total channel count
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None)

    # ── fMRI ──
    df_fmri = pd.read_pickle(fmri_path)
    fmri_np = df_fmri[[roi_col]].to_numpy().T          # (1, T_fmri)

    # low-pass filter fMRI
    fs = 1 / TR
    nyq = 0.5 * fs
    b, a = butter(N=5, Wn=0.15/nyq, btype='low', analog=False)
    fmri_np = filtfilt(b, a, fmri_np, axis=1)
    fmri_norm, _ = preproc.normalize_data(fmri_np)

    # ── Epoching ──
    data_epoch, _ = preproc.epoching_seq2one(
        raw, fmri_norm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)

    n = len(data_epoch["eeg"])
    traincrop = int(0.8 * n)
    valcrop   = int(0.1 * n) + traincrop
    t_overlap = 20
    N_overlap = math.ceil(t_overlap / TR)
    valcrop  += N_overlap

    eeg_test  = torch.stack([torch.tensor(x) for x in data_epoch["eeg"][valcrop:]])
    fmri_test = torch.stack([torch.tensor(x) for x in data_epoch["fmri"][valcrop:]]).squeeze()
    return eeg_test, fmri_test


def build_model(ckpt_path, init_values=0.1):
    model = create_model(
        'neurobolt_default',
        EEG_channel=26,
        num_roi=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,
        init_values=init_values,
        qkv_bias=True,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt

    # Remove keys that might mismatch
    for k in ['head.weight', 'head.bias']:
        if k in state and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    all_keys = list(state.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state.pop(key)

    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def infer(model, eeg_tensor, batch_size=64):
    """Run batched inference and return predictions as numpy array."""
    preds = []
    for i in range(0, len(eeg_tensor), batch_size):
        batch = eeg_tensor[i:i+batch_size].float().to(DEVICE) / 100.0
        batch = rearrange(batch, 'B N (A T) -> B N A T', T=200)
        input_chans = get_input_chans(CH_NAMES)
        out = model(batch, input_chans=input_chans)
        preds.append(out.cpu().squeeze(-1))
    return torch.cat(preds).numpy()


def evaluate_roi(roi_col, ckpt_path, data_root, all_scans):
    """Evaluate one ROI checkpoint over all test scans → R, MSE."""
    print(f'  Loading model from {ckpt_path}...', flush=True)
    model = build_model(ckpt_path)

    all_pred, all_true = [], []
    per_scan_r = []

    for (sub_idx, scan_idx) in all_scans:
        try:
            eeg_t, fmri_t = load_scan(sub_idx, scan_idx, data_root, roi_col)
            pred = infer(model, eeg_t)
            true = fmri_t.numpy() if hasattr(fmri_t, 'numpy') else np.array(fmri_t)

            if len(pred) < 5:
                continue
            r, _ = pearsonr(pred, true)
            mse   = np.mean((pred - true) ** 2)
            per_scan_r.append(r)
            all_pred.append(pred)
            all_true.append(true)
            print(f'    sub{sub_idx:02d}-scan{scan_idx:02d}: R={r:.3f}, MSE={mse:.3f}')
        except Exception as e:
            print(f'    sub{sub_idx:02d}-scan{scan_idx:02d}: SKIP ({e})')

    if not all_pred:
        return float('nan'), float('nan')

    all_pred_cat = np.concatenate(all_pred)
    all_true_cat = np.concatenate(all_true)
    r_global, _  = pearsonr(all_pred_cat, all_true_cat)
    mse_global   = np.mean((all_pred_cat - all_true_cat) ** 2)
    print(f'  → Global R={r_global:.3f}, MSE={mse_global:.3f} '
          f'(mean per-scan R={np.mean(per_scan_r):.3f})')
    return r_global, mse_global


def print_table(results):
    """Print results in NeuroBOLT Table 1 format."""
    headers = ["Method"] + [r[2] for r in ROIS] + ["Avg. R"]
    divider = "─" * (16 + 17 * len(ROIS) + 10)
    print()
    print("=" * len(divider))
    print("  NeuroBOLT Reproduction — Inter-Subject Evaluation (Pearson R / MSE)")
    print("=" * len(divider))
    # Header
    h = f"{'Method':<16}" + "".join(f"{h:>17}" for h in headers[1:])
    print(h)
    print("-" * len(h))
    for method, row in results.items():
        r_vals  = [row[roi[2]]['R']   for roi in ROIS]
        mse_vals= [row[roi[2]]['MSE'] for roi in ROIS]
        avg_r   = row.get('avg_r', float('nan'))
        line = f"{method:<16}"
        for r, m in zip(r_vals, mse_vals):
            cell = f"{r:.3f}/{m:.3f}"
            line += f"{cell:>17}"
        line += f"{avg_r:>10.3f}"
        print(line)
    print("=" * len(h))
    print()
    print("Note: Per-cell format is R/MSE. Avg.R = mean Pearson R across all 7 displayed ROIs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/data')
    parser.add_argument('--ckpt_dir',  default='C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/code/checkpoints')
    parser.add_argument('--mode', default='intra', choices=['intra'],
                        help='intra: use all 29 scans (80/10/10 per-scan split)')
    args = parser.parse_args()

    # All 29 available scans
    all_scans = [
        (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
        (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
        (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
        (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
    ]

    results = {'NeuroBOLT': {}}
    print(f'\n{"="*60}')
    print(f'Evaluating NeuroBOLT pretrained checkpoints (intra-subject)')
    print(f'{"="*60}')
    print(f'Data root : {args.data_root}')
    print(f'Device    : {DEVICE}')
    print(f'Scans     : {len(all_scans)}')
    print()

    r_all = []
    for roi_col, ckpt_fname, display in ROIS:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_fname)
        print(f'\n[ROI] {display} ({roi_col})')
        r, mse = evaluate_roi(roi_col, ckpt_path, args.data_root, all_scans)
        results['NeuroBOLT'][display] = {'R': r, 'MSE': mse}
        r_all.append(r)

    results['NeuroBOLT']['avg_r'] = float(np.nanmean(r_all))

    # Print table
    print_table(results)

    # Save JSON
    import json
    out_path = os.path.join(os.path.dirname(__file__), 'neurobolt_intra_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            k: {rk: {mk: float(mv) for mk, mv in rv.items()} if isinstance(rv, dict) else float(rv)
                for rk, rv in v.items()}
            for k, v in results.items()
        }, f, indent=2)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
