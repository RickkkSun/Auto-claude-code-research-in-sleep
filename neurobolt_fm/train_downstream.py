"""
train_downstream.py — EEG-fMRI Pairing Test (downstream).

TASK: Given a (EEG_features, fMRI) pair, predict whether it is "matched"
(truly corresponding in time) or "mismatched" (shuffled).

Rationale: Real EEG and real fMRI are strongly coupled via neurovascular
coupling. A linear classifier can learn this coupling and easily distinguish
matched pairs from shuffled pairs → real baseline ~0.90-0.99.

This tests whether GENERATED fMRI preserves EEG-fMRI conditioning:
  - If synth fMRI actually depends on input EEG → (EEG_i, synth_fMRI_i) has
    coupling, (EEG_i, synth_fMRI_j) does not → classifier works
  - If synth ignores EEG and generates "average" fMRI → no coupling →
    classifier near chance

Setup:
  - Positive pair (label=1): (EEG_i, fMRI_i) — matched
  - Negative pair (label=0): (EEG_i, fMRI_j) j≠i — shuffled within scan
  - Features: concat [EEG_feat (200), fMRI (7)] → 207 dim
  - Classifier: Logistic Regression (classical, no class reweighting)

Usage:
  python train_downstream.py --head cfm --dataset neurobolt
"""

import sys, os, json, argparse, warnings, re
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'code'))

import train_comparison_ddpm as _orig
import train_comparison_ddpm_external as _ext
from generative_heads import HEAD_REGISTRY, HEAD_DISPLAY, ALL_HEADS

DEVICE = _orig.DEVICE
N_ROIS = _orig.N_ROIS
EEG_FEAT_DIM = 200

DOWN_DIR = os.path.join(BASE, 'external_results', 'downstream')
FULL_LOAD_DATASETS = {'natview-monkey1_run-01'}


# ─── Utilities ──────────────────────────────────────────────────────────────

def acc_with_wilson_ci(y_true, y_pred, alpha=0.05):
    """Classical accuracy with Wilson 95% CI."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_correct = int((y_true == y_pred).sum())
    n_total = len(y_true)
    if n_total == 0:
        return np.nan, (np.nan, np.nan), 0, 0
    acc = n_correct / n_total
    ci_low, ci_high = proportion_confint(n_correct, n_total, alpha=alpha, method='wilson')
    return float(acc), (float(ci_low), float(ci_high)), n_correct, n_total


# ─── Pair sample construction ──────────────────────────────────────────────

def pair_feature_window(pred_window, fmri_window):
    """
    pred_window: [W, 7] Ridge-predicted fMRI for a window
    fmri_window: [W, 7] candidate fMRI window
    Feature = [corr_per_ROI (7), mean_abs_diff (7), mean_pred (7), mean_fmri (7),
               flat_correlation (1)] = 29 dim.
    Positive (matched) pair: high corr, small diff.
    Negative (shuffled) pair: low corr, larger diff.
    """
    # Per-ROI Pearson r across time
    corrs = []
    for j in range(pred_window.shape[1]):
        p, f = pred_window[:, j], fmri_window[:, j]
        ps, fs = p.std(), f.std()
        if ps < 1e-8 or fs < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(p, f)[0, 1]))
    corrs = np.array(corrs, dtype=np.float32)
    mean_abs_diff = np.abs(pred_window - fmri_window).mean(0)
    mean_pred = pred_window.mean(0)
    mean_fmri = fmri_window.mean(0)
    # Global correlation across all ROI-timepoints
    p_flat = pred_window.flatten()
    f_flat = fmri_window.flatten()
    if p_flat.std() < 1e-8 or f_flat.std() < 1e-8:
        flat_corr = 0.0
    else:
        flat_corr = float(np.corrcoef(p_flat, f_flat)[0, 1])
    return np.concatenate([corrs, mean_abs_diff, mean_pred, mean_fmri,
                            np.array([flat_corr])]).astype(np.float32)


WINDOW_SIZE = 60  # timepoints per window for pairing features (~120s at TR=2s)


def build_real_pairs(feat_list, fmri_list, ridge, seed, window_size=WINDOW_SIZE, stride=5):
    """
    Windowed pairing: each window of ridge-predicted fMRI is paired with
    either the matching real fMRI window (positive) or a cross-scan random
    fMRI window (negative). Correlation-based features expose matching.
    """
    rng = np.random.default_rng(seed)
    all_pred, all_fmri = [], []
    for feat, fmri in zip(feat_list, fmri_list):
        fe = feat.numpy() if hasattr(feat, 'numpy') else feat
        fm = fmri.numpy() if hasattr(fmri, 'numpy') else fmri
        all_pred.append(ridge.predict(fe))
        all_fmri.append(fm)

    n_scans = len(all_pred)
    all_X, all_y = [], []
    for s, (pred, fm) in enumerate(zip(all_pred, all_fmri)):
        T = len(pred)
        if T < window_size: continue
        for start in range(0, T - window_size + 1, stride):
            pw = pred[start:start + window_size]
            fw = fm[start:start + window_size]
            all_X.append(pair_feature_window(pw, fw))
            all_y.append(1)
            # negative: cross-scan random window
            other_s = rng.integers(0, n_scans - 1) if n_scans > 1 else s
            if other_s >= s and n_scans > 1: other_s += 1
            other_fm = all_fmri[other_s]
            if len(other_fm) < window_size: continue
            o_start = rng.integers(0, len(other_fm) - window_size + 1)
            other_fw = other_fm[o_start:o_start + window_size]
            all_X.append(pair_feature_window(pw, other_fw))
            all_y.append(0)
    if not all_X:
        return np.empty((0, 29), dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(all_X), np.array(all_y, dtype=int)


def build_synth_pairs(feat_list, synth_list_per_scan, ridge, seed,
                        window_size=WINDOW_SIZE, stride=5):
    """Same windowed approach but using synth fMRI."""
    rng = np.random.default_rng(seed)
    all_pred, all_synth = [], []
    for feat, synth in zip(feat_list, synth_list_per_scan):
        fe = feat.numpy() if hasattr(feat, 'numpy') else feat
        if hasattr(synth, 'numpy'):
            sm = synth.mean(0).numpy()
        else:
            sm = synth.mean(axis=0)
        all_pred.append(ridge.predict(fe))
        all_synth.append(sm)

    n_scans = len(all_pred)
    all_X, all_y = [], []
    for s, (pred, sm) in enumerate(zip(all_pred, all_synth)):
        T = len(pred)
        if T < window_size: continue
        for start in range(0, T - window_size + 1, stride):
            pw = pred[start:start + window_size]
            fw = sm[start:start + window_size]
            all_X.append(pair_feature_window(pw, fw))
            all_y.append(1)
            other_s = rng.integers(0, n_scans - 1) if n_scans > 1 else s
            if other_s >= s and n_scans > 1: other_s += 1
            other_sm = all_synth[other_s]
            if len(other_sm) < window_size: continue
            o_start = rng.integers(0, len(other_sm) - window_size + 1)
            other_fw = other_sm[o_start:o_start + window_size]
            all_X.append(pair_feature_window(pw, other_fw))
            all_y.append(0)
    if not all_X:
        return np.empty((0, 29), dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(all_X), np.array(all_y, dtype=int)


# ─── Full-scan loader for short datasets ────────────────────────────────────

def load_full_scans(dataset_key):
    import math
    import mne
    import pandas as pd
    from scipy.signal import butter, filtfilt
    from dataset_maker import preproc as _p

    data_root = os.path.join(BASE, 'external_processed', dataset_key)
    tr = _ext.EXTERNAL_DATASET_CONFIGS.get(dataset_key, {}).get('tr', 2.1)
    eeg_dir = os.path.join(data_root, 'EEG')
    scans = sorted(f.replace('_eeg.set', '')
                    for f in os.listdir(eeg_dir) if f.endswith('_eeg.set'))

    CH_NAMES = _orig.CH_NAMES
    TMIN, CROP, EVENT = _orig.TMIN, _orig.CROP, _orig.EVENT
    ROI_COLS = _orig.ROI_COLS

    eeg_out, fmri_out, scan_out = [], [], []
    for pat in scans:
        eeg_path = os.path.join(data_root, 'EEG', f'{pat}_eeg.set')
        fm_path = os.path.join(data_root, 'fMRI_difumo64', f'{pat}_difumo64_roi.pkl')
        if not os.path.exists(eeg_path) or not os.path.exists(fm_path):
            continue
        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
            extra = [ch for ch in raw.ch_names if ch not in CH_NAMES]
            if extra: raw.drop_channels(extra, on_missing='ignore')
            if raw.info['sfreq'] != 200: raw.resample(200)
            raw.load_data()
            raw.filter(l_freq=0.5, h_freq=None, verbose=False)
            df = pd.read_pickle(fm_path)
            b_lp, a_lp = butter(5, 0.15 / (0.5 / tr), btype='low')
            eeg_all = None
            all_fmri = []
            skip = False
            for col in ROI_COLS:
                if col not in df.columns:
                    skip = True; break
                fm = df[[col]].to_numpy().T
                fm = filtfilt(b_lp, a_lp, fm, axis=1)
                fm, _ = _p.normalize_data(fm)
                ep, _ = _p.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
                if eeg_all is None:
                    eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32)
                                            for x in ep['eeg']])
                all_fmri.append(torch.tensor(
                    [float(np.asarray(x).flat[0]) for x in ep['fmri']],
                    dtype=torch.float32))
            if skip or eeg_all is None:
                continue
            fmri_all = torch.stack(all_fmri, dim=1)
            eeg_out.append(eeg_all); fmri_out.append(fmri_all); scan_out.append(pat)
        except Exception as e:
            print(f'  {pat}: SKIP ({e})')
    return eeg_out, fmri_out, scan_out


# ─── Generative head helpers ────────────────────────────────────────────────

def _train_ddpm_wrapper(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
                        epochs=300, lr=3e-4, bs=256, device='cuda'):
    from generative_heads import HeadWrapper
    ddpm = _orig.train_ddpm(feat_tr, fmri_tr, feat_val, fmri_val,
                             feat_dim=feat_dim, epochs=epochs, lr=lr, bs=bs, device=device)
    def sample_fn(features, n_samples, bs_):
        ddpm.denoiser.eval()
        return ddpm.ddim_sample(features.to(device), n_samples=n_samples,
                                ddim_steps=50, bs=bs_)
    return HeadWrapper(sample_fn, ddpm.mu_tgt, ddpm.sig_tgt)


def get_train_fn(head):
    return _train_ddpm_wrapper if head == 'ddpm' else HEAD_REGISTRY[head]


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run(head_name, dataset_key, epochs=300, feat_dim=200, n_synth=10):
    print(f'\n{"="*60}')
    print(f'  EEG-fMRI Pairing: {HEAD_DISPLAY[head_name]} on {dataset_key}')
    print(f'{"="*60}\n')

    # ── Load data ──────────────────────────────────────────────────
    full_load = dataset_key in FULL_LOAD_DATASETS
    if full_load:
        print(f'  [FULL-LOAD]')
        eeg_list, fmri_list, scan_names = load_full_scans(dataset_key)
        eeg_tv, fmri_tv, eeg_te, fmri_te = [], [], [], []
        for e, f in zip(eeg_list, fmri_list):
            n = len(e)
            split = int(0.8 * n)
            eeg_tv.append(e[:split]); fmri_tv.append(f[:split])
            eeg_te.append(e[split:]); fmri_te.append(f[split:])
    elif dataset_key == 'neurobolt':
        eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names = _orig.load_all_scans()
    else:
        data_root = os.path.join(BASE, 'external_processed', dataset_key)
        tr_val = _ext.EXTERNAL_DATASET_CONFIGS.get(dataset_key, {}).get('tr', 2.1)
        eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names = _ext.load_all_scans_ext(data_root, tr_val)

    print(f'  Loaded {len(scan_names)} scans')

    # ── Extract EEG features (shared across methods) ───────────────
    backbone = _orig._make_backbone('neurobolt', feat_dim)
    print('  Extracting EEG features...')
    feat_tv_list = _orig.extract_features(backbone, eeg_tv, device=DEVICE)
    feat_te_list = _orig.extract_features(backbone, eeg_te, device=DEVICE)

    feat_tv_all = torch.cat(feat_tv_list)
    fmri_tv_all = torch.cat(fmri_tv)

    n_tv = len(feat_tv_all)
    split = int(0.8 / 0.9 * n_tv)
    feat_tr, fmri_tr = feat_tv_all[:split], fmri_tv_all[:split]
    feat_val, fmri_val = feat_tv_all[split:], fmri_tv_all[split:]

    print(f'  EEG Features: train={len(feat_tr)}, val={len(feat_val)}')

    # ── Train generative head ──────────────────────────────────────
    train_fn = get_train_fn(head_name)
    head = train_fn(feat_tr, fmri_tr, feat_val, fmri_val,
                     feat_dim=feat_dim, epochs=epochs, device=DEVICE)

    # ── Generate synth per scan ────────────────────────────────────
    print(f'  Generating {n_synth} synth samples per scan...')
    synth_tv_per_scan = []
    for feat_scan in feat_tv_list:
        with torch.no_grad():
            samples_n = head.sample(feat_scan, n_samples=n_synth, bs=256)
            samples = samples_n * head.sig_tgt + head.mu_tgt   # [K, T, 7]
        synth_tv_per_scan.append(samples)

    # ── Train Ridge EEG→fMRI (shared feature extractor for pairing) ──
    print('  Training Ridge EEG→fMRI for pairing features...')
    ridge = Ridge(alpha=1.0).fit(feat_tr.numpy(), fmri_tr.numpy())

    # ── Build pair samples ─────────────────────────────────────────
    print('  Building EEG-fMRI pairs...')
    X_real_all, y_real_all = build_real_pairs(feat_tv_list, fmri_tv, ridge, seed=0)
    X_tr_synth, y_tr_synth = build_synth_pairs(feat_tv_list, synth_tv_per_scan, ridge, seed=42)

    if len(X_real_all) < 20:
        print(f'  SKIP: only {len(X_real_all)} real pairs')
        return {'head': head_name, 'dataset': dataset_key,
                'subject_id': None, 'error': 'insufficient_pairs'}

    # 80/20 pair-level split (keep positive+negative balanced)
    rng = np.random.default_rng(7)
    n_pairs = len(X_real_all) // 2   # each "pair" = positive+negative at same i
    pair_idx = rng.permutation(n_pairs)
    split_pt = int(0.8 * n_pairs)
    tr_pairs, te_pairs = pair_idx[:split_pt], pair_idx[split_pt:]
    tr_sample_idx = np.concatenate([2*tr_pairs, 2*tr_pairs + 1])
    te_sample_idx = np.concatenate([2*te_pairs, 2*te_pairs + 1])
    X_tr_real = X_real_all[tr_sample_idx]
    y_tr_real = y_real_all[tr_sample_idx]
    X_te_real = X_real_all[te_sample_idx]
    y_te_real = y_real_all[te_sample_idx]

    print(f'  Samples: train_real={len(X_tr_real)}, train_synth={len(X_tr_synth)}, '
          f'test={len(X_te_real)}')

    # ── Train classifiers ─────────────────────────────────────────
    scaler = StandardScaler().fit(X_tr_real)
    X_tr_r = scaler.transform(X_tr_real)
    X_te_r = scaler.transform(X_te_real)
    X_tr_s = scaler.transform(X_tr_synth) if len(X_tr_synth) > 0 else X_tr_synth

    def _fit_and_eval(Xtr, ytr):
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
            y_pred = clf.predict(X_te_r)
            return acc_with_wilson_ci(y_te_real, y_pred)
        except Exception as e:
            print(f'  fit error: {e}')
            return np.nan, (np.nan, np.nan), 0, len(y_te_real)

    acc_real, ci_real, nc_real, n_te = _fit_and_eval(X_tr_r, y_tr_real)
    if len(X_tr_s) > 0:
        acc_synth, ci_synth, nc_synth, _ = _fit_and_eval(X_tr_s, y_tr_synth)
    else:
        acc_synth, ci_synth, nc_synth = np.nan, (np.nan, np.nan), 0

    result = {
        'head': head_name,
        'dataset': dataset_key,
        'task': 'eeg_fmri_pairing',
        'chance_level': 0.5,
        'subject_id': {   # key name kept for table-gen compatibility
            'acc_real': acc_real,
            'ci_real': list(ci_real),
            'n_correct_real': nc_real,
            'acc_synth': acc_synth,
            'ci_synth': list(ci_synth),
            'n_correct_synth': nc_synth,
            'n_test_total': int(n_te),
            'n_real_train': int(len(X_tr_r)),
            'n_synth_total': int(len(X_tr_s)),
            'chance': 0.5,
        },
    }
    print(f'  acc_real ={acc_real:.3f} [{ci_real[0]:.3f}, {ci_real[1]:.3f}]  '
          f'({nc_real}/{n_te})')
    print(f'  acc_synth={acc_synth:.3f} [{ci_synth[0]:.3f}, {ci_synth[1]:.3f}]  '
          f'({nc_synth}/{n_te})')
    return result


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', required=True, choices=ALL_HEADS)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n_synth', type=int, default=10)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    result = run(args.head, args.dataset, args.epochs, n_synth=args.n_synth)

    os.makedirs(DOWN_DIR, exist_ok=True)
    out_path = args.out or os.path.join(DOWN_DIR, f'{args.dataset}_downstream.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing[args.head] = result
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults → {out_path}')
