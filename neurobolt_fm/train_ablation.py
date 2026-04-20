"""
train_ablation.py — Generative head ablation study.

Fixed backbone: NeuroBOLT (single Cuneus checkpoint, 200D features).
Variable: 8 generative heads (DDPM, EDM, Score SDE, Consistency, CVAE, cGAN,
          Rectified Flow, CFM).

Supports both original NeuroBOLT dataset and external preprocessed datasets.

Usage:
  python train_ablation.py --head edm --mode intra --dataset neurobolt
  python train_ablation.py --head cvae --mode pooled --dataset ds003768
  python train_ablation.py --head cfm --mode intra --dataset ds002336
"""

import sys, os, gc, json, math, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'code'))

# Import from existing modules
import train_comparison_ddpm as _orig
from generative_heads import HEAD_REGISTRY, HEAD_DISPLAY, ALL_HEADS

DEVICE = _orig.DEVICE
N_ROIS = _orig.N_ROIS
N_CHAN  = _orig.N_CHAN
SEQ_LEN = _orig.SEQ_LEN

# Also import external data loader
import train_comparison_ddpm_external as _ext

ABL_DIR = os.path.join(BASE, 'external_results', 'ablation')

# ─── Evaluation (reuse from _orig) ───────────────────────────────────────────

from scipy.stats import pearsonr


def evaluate_head(head, feat_te, fmri_te, n_samples=50, device=DEVICE):
    """Evaluate any HeadWrapper. Returns dict with R, CRPS, FC-MAE, CI."""
    samples_n = head.sample(feat_te, n_samples=n_samples, bs=256)  # [K, N, 7] normalized CPU
    samples = samples_n * head.sig_tgt + head.mu_tgt  # un-normalize
    pred_mean = samples.mean(0)

    roi_r = []
    for j in range(N_ROIS):
        r, _ = pearsonr(pred_mean[:, j].numpy(), fmri_te[:, j].numpy())
        roi_r.append(float(r))
    avg_r = float(np.nanmean(roi_r))

    crps   = _orig.compute_crps(samples, fmri_te)
    fc_mae = _orig.compute_fc_mae(samples, fmri_te)

    # Bootstrap CI
    ci = _orig.bootstrap_ci(samples, fmri_te)

    return {
        'avg_r': avg_r, 'roi_r': roi_r,
        'crps': crps, 'fc_mae': fc_mae,
        'crps_ci': ci['crps_ci'], 'r_ci': ci['r_ci'],
    }


def _train_ddpm_wrapper(feat_tr, fmri_tr, feat_val, fmri_val, feat_dim=200,
                         epochs=300, lr=3e-4, bs=256, device='cuda'):
    """Wrap existing DDPM to return HeadWrapper interface."""
    from generative_heads import HeadWrapper
    ddpm = _orig.train_ddpm(feat_tr, fmri_tr, feat_val, fmri_val,
                            feat_dim=feat_dim, epochs=epochs, lr=lr, bs=bs, device=device)

    def sample_fn(features, n_samples, bs_):
        ddpm.denoiser.eval()
        return ddpm.ddim_sample(features.to(device), n_samples=n_samples,
                                ddim_steps=50, bs=bs_)

    return HeadWrapper(sample_fn, ddpm.mu_tgt, ddpm.sig_tgt)


# ─── Pipeline ────────────────────────────────────────────────────────────────

def get_train_fn(head_name):
    if head_name == 'ddpm':
        return _train_ddpm_wrapper
    return HEAD_REGISTRY[head_name]


def run_intra(head_name, data_loader_fn, feat_dim=200, epochs=300,
              neurobolt_ckpt=None):
    """Intra-subject ablation: per-scan NeuroBOLT features → head."""
    print(f'\n=== INTRA-SUBJECT: NeuroBOLT + {HEAD_DISPLAY[head_name]} ===\n')

    eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names = data_loader_fn()

    # Load shared NeuroBOLT backbone (frozen)
    backbone = _orig._make_backbone('neurobolt', feat_dim, neurobolt_ckpt=neurobolt_ckpt)
    train_fn = get_train_fn(head_name)

    scan_results = []
    for i, scan in enumerate(scan_names):
        print(f'\n  [Scan {i+1}/{len(scan_names)}] {scan}', flush=True)

        # Extract features
        n_tv = len(eeg_tv[i])
        split = int(0.8 / 0.9 * n_tv)
        feat_all = _orig.extract_features(backbone, [eeg_tv[i]], device=DEVICE)[0]
        feat_tr, fmri_tr   = feat_all[:split], fmri_tv[i][:split]
        feat_val, fmri_val = feat_all[split:], fmri_tv[i][split:]
        feat_tst = _orig.extract_features(backbone, [eeg_te[i]], device=DEVICE)[0]
        fmri_tst = fmri_te[i]

        # Train head
        head = train_fn(feat_tr, fmri_tr, feat_val, fmri_val,
                        feat_dim=feat_dim, epochs=epochs, device=DEVICE)

        # Evaluate
        res = evaluate_head(head, feat_tst, fmri_tst, n_samples=50, device=DEVICE)
        print(f'    avg_R={res["avg_r"]:.4f}  CRPS={res["crps"]:.4f}  FC-MAE={res["fc_mae"]:.4f}')
        scan_results.append({'scan': scan, **res})

        del head; gc.collect(); torch.cuda.empty_cache()

    # Aggregate
    all_r    = [r['avg_r']  for r in scan_results]
    all_crps = [r['crps']   for r in scan_results]
    all_fc   = [r['fc_mae'] for r in scan_results]
    roi_r_mat = np.array([r['roi_r'] for r in scan_results])
    S = len(scan_results)

    rng = np.random.default_rng(42)
    crps_boot, r_boot = [], []
    for _ in range(500):
        idx = rng.integers(0, S, size=S)
        crps_boot.append(float(np.array(all_crps)[idx].mean()))
        r_boot.append(float(np.array(all_r)[idx].mean()))
    crps_ci = (float(np.quantile(crps_boot, 0.025)), float(np.quantile(crps_boot, 0.975)))
    r_ci    = (float(np.quantile(r_boot, 0.025)), float(np.quantile(r_boot, 0.975)))

    summary = {
        'backbone': 'neurobolt', 'head': head_name, 'mode': 'intra',
        'avg_r': float(np.nanmean(all_r)),
        'avg_r_std': float(np.nanstd(all_r)),
        'r_ci': r_ci,
        'roi_r': np.nanmean(roi_r_mat, axis=0).tolist(),
        'roi_r_mean': np.nanmean(roi_r_mat, axis=0).tolist(),
        'crps': float(np.mean(all_crps)),
        'crps_ci': crps_ci,
        'fc_mae': float(np.mean(all_fc)),
        'per_scan': scan_results,
    }
    print(f'\n  INTRA SUMMARY: avg_R={summary["avg_r"]:.4f}±{summary["avg_r_std"]:.4f}'
          f'  CRPS={summary["crps"]:.4f}  FC-MAE={summary["fc_mae"]:.4f}')
    return summary


def run_pooled(head_name, data_loader_fn, feat_dim=200, epochs=300,
               neurobolt_ckpt=None):
    """Pooled (inter-subject) ablation."""
    print(f'\n=== POOLED: NeuroBOLT + {HEAD_DISPLAY[head_name]} ===\n')

    eeg_tv, fmri_tv, eeg_te, fmri_te, scan_names = data_loader_fn()

    backbone = _orig._make_backbone('neurobolt', feat_dim, neurobolt_ckpt=neurobolt_ckpt)
    train_fn = get_train_fn(head_name)

    # Extract features
    print('  Extracting features...', flush=True)
    feat_tv_list = _orig.extract_features(backbone, eeg_tv, device=DEVICE)
    feat_te_list = _orig.extract_features(backbone, eeg_te, device=DEVICE)

    feat_tv_all = torch.cat(feat_tv_list)
    fmri_tv_all = torch.cat(fmri_tv)
    feat_te_all = torch.cat(feat_te_list)
    fmri_te_all = torch.cat(fmri_te)

    n_tv = len(feat_tv_all)
    split = int(0.8 / 0.9 * n_tv)
    feat_tr, fmri_tr   = feat_tv_all[:split], fmri_tv_all[:split]
    feat_val, fmri_val = feat_tv_all[split:], fmri_tv_all[split:]

    print(f'  pooled train={len(feat_tr)}, val={len(feat_val)}, test={len(feat_te_all)}')

    head = train_fn(feat_tr, fmri_tr, feat_val, fmri_val,
                    feat_dim=feat_dim, epochs=epochs, device=DEVICE)

    res = evaluate_head(head, feat_te_all, fmri_te_all, n_samples=50, device=DEVICE)

    summary = {
        'backbone': 'neurobolt', 'head': head_name, 'mode': 'pooled',
        **res,
    }
    print(f'\n  POOLED SUMMARY: avg_R={res["avg_r"]:.4f}  CRPS={res["crps"]:.4f}'
          f'  FC-MAE={res["fc_mae"]:.4f}')
    return summary


# ─── Data loader factory ─────────────────────────────────────────────────────

def make_data_loader(dataset_key):
    """Return a callable that loads data for the given dataset."""
    if dataset_key == 'neurobolt':
        return _orig.load_all_scans

    # External dataset
    data_root = os.path.join(BASE, 'external_processed', dataset_key)
    tr = _ext.EXTERNAL_DATASET_CONFIGS.get(dataset_key, {}).get('tr', 2.1)

    def loader():
        return _ext.load_all_scans_ext(data_root, tr)

    return loader


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative head ablation study')
    parser.add_argument('--head', required=True, choices=ALL_HEADS)
    parser.add_argument('--mode', default='intra', choices=['intra', 'pooled'])
    parser.add_argument('--dataset', required=True,
                        help='neurobolt | ds003768 | ds005795 | ds006040 | ds002336 | '
                             'ds002725 | natview-monkey1_run-01 | ds007216')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--out', default=None)
    parser.add_argument('--neurobolt_ckpt', default=None)
    args = parser.parse_args()

    print(f'Device:  {DEVICE}')
    print(f'Dataset: {args.dataset}  Head: {args.head}  Mode: {args.mode}')
    print(f'Epochs:  {args.epochs}')

    loader = make_data_loader(args.dataset)
    feat_dim = 200  # NeuroBOLT output dim

    if args.mode == 'intra':
        result = run_intra(args.head, loader, feat_dim=feat_dim,
                           epochs=args.epochs, neurobolt_ckpt=args.neurobolt_ckpt)
    else:
        result = run_pooled(args.head, loader, feat_dim=feat_dim,
                            epochs=args.epochs, neurobolt_ckpt=args.neurobolt_ckpt)

    # Save
    os.makedirs(ABL_DIR, exist_ok=True)
    out_path = args.out or os.path.join(ABL_DIR, f'{args.dataset}_ablation.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    key = f'{args.head}_{args.mode}'
    existing[key] = result
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nResults → {out_path}  [{key}]')

    roi_r = result.get('roi_r_mean') or result.get('roi_r', [])
    rnames = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
    for name, r in zip(rnames, roi_r):
        print(f'  {name:<14}: R={r:.4f}')
