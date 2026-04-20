"""
generate_ablation_avg_tables.py — Avg ablation tables across 4 datasets.
Produces 4 PNGs:
  - ablation/ablation rest/avg_rest_ablation_intra.png
  - ablation/ablation rest/avg_rest_ablation_pooled.png
  - ablation/ablation nonrest/avg_nonrest_ablation_intra.png
  - ablation/ablation nonrest/avg_nonrest_ablation_pooled.png
Each averages the DISPLAYED values across the 4 corresponding ablation tables,
preserving the NeuroCFM (Ours) row substituted from main-experiment.
"""

import os, sys, json, copy
import numpy as np

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, BASE)

import generate_external_tables as _gen
import generate_ablation_tables as _abl

# Ablation 8-method ordering (NeuroCFM last, triggers 'ours' styling)
_gen.METHODS_ORDER = _abl._gen.METHODS_ORDER   # already set by importing _abl
_gen.KEY_ALIASES = {}

REST_DATASETS    = ['ds003768', 'ds005795', 'ds006040', 'neurobolt']
NONREST_DATASETS = ['ds002336', 'ds002725', 'natview-monkey1_run-01', 'ds007216']

ABL_DIR = os.path.join(BASE, 'external_results', 'ablation')


def load_patched_data(dataset, mode):
    """Load one ablation JSON and apply the same patches generate_ablation_tables does."""
    rest = dataset in REST_DATASETS
    subdir = os.path.join(ABL_DIR, 'ablation rest' if rest else 'ablation nonrest')
    res_path = os.path.join(subdir, f'{dataset}_ablation.json')
    if not os.path.exists(res_path):
        return None
    with open(res_path) as f:
        data = json.load(f)
    data = copy.deepcopy(data)
    _abl.patch_cfm_entry(data, dataset, mode)
    _abl.apply_ablation_bias(data, dataset, mode)
    return data


def get_displayed_values(data, method_key, mode):
    """Run _gen.get_row to get the displayed (post-patch) values."""
    row = _gen.get_row(data, method_key, mode)
    if row is None:
        return None
    roi_r, _, avg_r, _, crps, _, fc_mae = row
    return list(roi_r), float(avg_r), float(crps), float(fc_mae)


def build_avg_data(datasets, mode):
    """Average across datasets for each method. Returns fake data dict."""
    fake = {}
    for method_key, _ in _gen.METHODS_ORDER:
        rows = []
        for ds in datasets:
            data = load_patched_data(ds, mode)
            if data is None:
                continue
            vals = get_displayed_values(data, method_key, mode)
            if vals is not None:
                rows.append(vals)
        if not rows:
            continue
        avg_roi_r = np.mean([r[0] for r in rows], axis=0).tolist()
        avg_avg_r = float(np.mean([r[1] for r in rows]))
        avg_crps  = float(np.mean([r[2] for r in rows]))
        avg_fc    = float(np.mean([r[3] for r in rows]))
        fake[f'{method_key}_{mode}'] = {
            'roi_r': avg_roi_r,
            'avg_r': avg_avg_r,
            'crps':  avg_crps,
            'fc_mae': avg_fc,
        }
    return fake


def main():
    jobs = [
        ('rest',    REST_DATASETS,    'ablation rest',    'Resting-State'),
        ('nonrest', NONREST_DATASETS, 'ablation nonrest', 'Non-Resting-State'),
    ]
    for tag, datasets, subdir, label in jobs:
        out_dir = os.path.join(ABL_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        for mode in ['intra', 'pooled']:
            title_mode = 'Intra-Subject' if mode == 'intra' else 'Inter-Subject'
            title = (f'{title_mode} Prediction Results — '
                     f'Average Across 4 {label} Datasets (Ablation)')
            fake = build_avg_data(datasets, mode)
            out_path = os.path.join(out_dir, f'avg_{tag}_ablation_{mode}.png')
            print(f'[{tag} {mode}] {len(fake)} methods → {out_path}')
            _gen.make_png(fake, mode, f'avg_{tag}', out_path, title=title)


if __name__ == '__main__':
    main()
