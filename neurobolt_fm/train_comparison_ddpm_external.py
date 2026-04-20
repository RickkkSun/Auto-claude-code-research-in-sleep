"""
train_comparison_ddpm_external.py — Run the full comparison benchmark on an
external preprocessed dataset (ds003768, ds005795, noddi, ds006040).

Preprocessed layout expected in external_processed/{dataset}/:
  EEG/          {sub}-scan01_eeg.set
  fMRI_difumo64/ {sub}-scan01_difumo64_roi.pkl

Usage:
  python train_comparison_ddpm_external.py \
      --backbone sparc --mode intra --dataset ds003768

  python train_comparison_ddpm_external.py \
      --backbone neurobolt --mode pooled --dataset ds005795 \
      --out external_results/ds005795_results.json
"""

import os, sys, json, math, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import mne
mne.set_log_level('WARNING')
from scipy.signal import butter, filtfilt

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, BASE)

# ── Import original module for backbone / DDPM / training logic ──────────────
import train_comparison_ddpm as _orig

DEVICE   = _orig.DEVICE
ROI_COLS = _orig.ROI_COLS
CH_NAMES = _orig.CH_NAMES
TMIN     = _orig.TMIN
CROP     = _orig.CROP
EVENT    = _orig.EVENT   # 'R149'
SEQ_LEN  = _orig.SEQ_LEN
N_CHAN   = _orig.N_CHAN
BASE_ORIG = _orig.BASE

# ── External dataset TR configs ───────────────────────────────────────────────
EXTERNAL_DATASET_CONFIGS = {
    "ds003768":             {"tr": 2.1},
    "ds005795":             {"tr": 2.0},
    "noddi":                {"tr": 2.16},
    "ds006040":             {"tr": 2.0},
    # Non-resting-state datasets
    "ds002725":             {"tr": 2.0},   # music listening
    "ds002336":             {"tr": 2.0},   # motor neurofeedback
    "natview-monkey1_run-01": {"tr": 2.1}, # film viewing
    "natview-inscapes":     {"tr": 2.1},   # naturalistic audio
    "natview-dme_run-01":   {"tr": 2.1},   # mental effort
    "ds007216":             {"tr": 2.0},   # gradCPT sustained attention
}


# ── External data loaders ─────────────────────────────────────────────────────

def discover_scans(data_root: str):
    """List patient names from EEG/*.set files."""
    eeg_dir = os.path.join(data_root, 'EEG')
    if not os.path.isdir(eeg_dir):
        raise FileNotFoundError(f"EEG dir not found: {eeg_dir}")
    return sorted(f.replace('_eeg.set', '')
                  for f in os.listdir(eeg_dir)
                  if f.endswith('_eeg.set'))


def load_scan_ext(patient_name: str, data_root: str, tr: float):
    """
    Load one external scan.
    Returns (eeg_tv, fmri_tv, eeg_te, fmri_te) or None.
    """
    try:
        from dataset_maker import preproc as _p
    except ImportError:
        sys.path.insert(0, os.path.join(BASE, 'code'))
        from dataset_maker import preproc as _p

    eeg_path = os.path.join(data_root, 'EEG', f'{patient_name}_eeg.set')
    fm_path  = os.path.join(data_root, 'fMRI_difumo64', f'{patient_name}_difumo64_roi.pkl')

    if not os.path.exists(eeg_path) or not os.path.exists(fm_path):
        return None

    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    # Drop channels not in the 26-channel set (preprocessing may include extras)
    extra = [ch for ch in raw.ch_names if ch not in CH_NAMES]
    if extra:
        raw.drop_channels(extra, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    df = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15 / (0.5 / tr), btype='low')

    eeg_all  = None
    all_fmri = []

    for col in ROI_COLS:
        if col not in df.columns:
            del raw
            return None
        fm = df[[col]].to_numpy().T
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = _p.normalize_data(fm)
        ep, _ = _p.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        n = len(ep['eeg'])
        if eeg_all is None:
            eeg_all = torch.stack(
                [torch.tensor(x, dtype=torch.float32) for x in ep['eeg']])
        all_fmri.append(torch.tensor(
            [float(np.asarray(x).flat[0]) for x in ep['fmri']], dtype=torch.float32))

    del raw
    fmri_all = torch.stack(all_fmri, dim=1)  # (n, 7)

    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / tr)

    return (eeg_all[:valcrop], fmri_all[:valcrop],
            eeg_all[valcrop:], fmri_all[valcrop:])


def load_all_scans_ext(data_root: str, tr: float):
    """Load all scans from an external preprocessed dataset."""
    scan_names = discover_scans(data_root)
    print(f'\n[Data] {len(scan_names)} potential scans in {data_root}', flush=True)

    eeg_tv_list, fmri_tv_list = [], []
    eeg_te_list, fmri_te_list = [], []
    kept = []

    for pat in scan_names:
        print(f'  {pat}...', end=' ', flush=True)
        try:
            res = load_scan_ext(pat, data_root, tr)
            if res is None:
                print('SKIP (missing)', flush=True); continue
            et, tt, ete, tte = res
            if len(ete) < 5:
                print('SKIP (test too short)', flush=True); continue
            eeg_tv_list.append(et);  fmri_tv_list.append(tt)
            eeg_te_list.append(ete); fmri_te_list.append(tte)
            kept.append(pat)
            print(f'tr={len(et)} te={len(ete)}', flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'SKIP ({e})', flush=True)

    print(f'  Loaded {len(kept)} scans')
    return eeg_tv_list, fmri_tv_list, eeg_te_list, fmri_te_list, kept


# ── Monkey-patch + run ────────────────────────────────────────────────────────

def run_on_external(mode: str, backbone_name: str, feat_dim: int,
                    epochs_bb: int, epochs_ddpm: int,
                    data_root: str, tr: float,
                    neurobolt_ckpt=None):
    """
    Patch the original module's load_all_scans + TR, then call run_intra/run_pooled.
    """
    _backup_TR  = _orig.TR
    _backup_las = _orig.load_all_scans

    # Update module-level TR for low-pass filter in any residual calls
    _orig.TR = tr

    def _patched_las():
        return load_all_scans_ext(data_root, tr)

    _orig.load_all_scans = _patched_las

    try:
        if mode == 'intra':
            result = _orig.run_intra(backbone_name, feat_dim, epochs_bb, epochs_ddpm,
                                      neurobolt_ckpt=neurobolt_ckpt)
        else:
            result = _orig.run_pooled(backbone_name, feat_dim, epochs_bb, epochs_ddpm,
                                       neurobolt_ckpt=neurobolt_ckpt)
    finally:
        _orig.TR             = _backup_TR
        _orig.load_all_scans = _backup_las

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparison benchmark on external preprocessed datasets')
    parser.add_argument('--backbone', required=True,
                        choices=['cnn_trans','ffcl','stt_trans','biot','sparc','contrawr',
                                 'beira','li2024','neurobolt','labram','reve','brainomni'])
    parser.add_argument('--mode', default='intra', choices=['intra', 'pooled'])
    parser.add_argument('--dataset', required=True,
                        help='Dataset key: ds003768 | ds005795 | noddi | ds006040')
    parser.add_argument('--data_root', default=None,
                        help='Override path to preprocessed dataset root')
    parser.add_argument('--feat_dim',    type=int, default=256)
    parser.add_argument('--epochs_bb',   type=int, default=100)
    parser.add_argument('--epochs_ddpm', type=int, default=300)
    parser.add_argument('--out', default=None,
                        help='Output JSON. Default: external_results/{dataset}_results.json')
    parser.add_argument('--neurobolt_ckpt', default=None)
    args = parser.parse_args()

    ds_key    = args.dataset
    data_root = args.data_root or os.path.join(BASE, 'external_processed', ds_key)
    tr        = EXTERNAL_DATASET_CONFIGS.get(ds_key, {}).get('tr', 2.1)

    if not os.path.isdir(data_root):
        print(f'ERROR: data_root not found: {data_root}'); sys.exit(1)

    n_scans = len(discover_scans(data_root))
    print(f'Device:   {DEVICE}')
    print(f'Dataset:  {ds_key}  (TR={tr}s, {n_scans} scans)')
    print(f'Backbone: {args.backbone}  Mode: {args.mode}')
    print(f'Epochs:   bb={args.epochs_bb}  ddpm={args.epochs_ddpm}')

    # Pretrained models have fixed output dims
    _PRETRAINED_FEAT_DIMS = {'neurobolt': 200, 'reve': 512, 'brainomni': 512}
    eff_feat_dim = _PRETRAINED_FEAT_DIMS.get(args.backbone, args.feat_dim)

    result = run_on_external(
        mode=args.mode,
        backbone_name=args.backbone,
        feat_dim=eff_feat_dim,
        epochs_bb=args.epochs_bb,
        epochs_ddpm=args.epochs_ddpm,
        data_root=data_root,
        tr=tr,
        neurobolt_ckpt=args.neurobolt_ckpt,
    )

    # Save
    os.makedirs(os.path.join(BASE, 'external_results'), exist_ok=True)
    out_path = args.out or os.path.join(BASE, 'external_results', f'{ds_key}_results.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    key = f'{args.backbone}_{args.mode}'
    existing[key] = result
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults → {out_path}  [{key}]')

    roi_r = result.get('roi_r_mean') or result.get('roi_r', [])
    rnames = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
    for name, r in zip(rnames, roi_r):
        print(f'  {name:<14}: R={r:.4f}')
    print(f'  {"Avg":<14}: R={result["avg_r"]:.4f}  CRPS={result.get("crps","?")}  '
          f'FC-MAE={result.get("fc_mae","?")}')
