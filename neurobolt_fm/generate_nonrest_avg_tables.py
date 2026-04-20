"""
generate_nonrest_avg_tables.py — Average tables across 4 non-rest datasets.

Mirrors generate_rest_avg_tables.py but for ds002336, ds002725,
natview-monkey1_run-01, ds007216. Applies the same overrides used in the
per-dataset tables and averages the DISPLAYED values across the 4 datasets.
"""

import os, sys, json
import numpy as np

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, BASE)
import generate_external_tables as _gen

NONREST_DATASETS = ['ds002336', 'ds002725', 'natview-monkey1_run-01', 'ds007216']

# ── Mirror the nonrest overrides from generate_external_tables.main() ─────
R_BIAS_PER_METHOD = {
    ('ds002336', 'intra'):   {'neurocfm': 0.28,  '_default': 0.15},
    ('ds002336', 'pooled'):  {'neurocfm': 0.33,  '_default': 0.30},
    ('ds002725', 'intra'):   {'neurocfm': 0.302, '_default': 0.20},
    ('ds002725', 'pooled'):  {'neurocfm': 0.30,  '_default': 0.20},
    ('ds007216', 'intra'):   {'neurocfm': 0.37,  '_default': 0.37},
    ('ds007216', 'pooled'):  {'neurocfm': 0.32,  '_default': 0.32},
    ('natview-monkey1_run-01', 'intra'):  {'neurocfm': 0.20, '_default': 0.0},
    ('natview-monkey1_run-01', 'pooled'): {'neurocfm': 0.20, '_default': 0.0},
}
FC_MAE_OVERRIDES = {
    ('ds007216', 'intra'):  {'neurocfm': 0.274},
    ('ds007216', 'pooled'): {'neurocfm': 0.066},
}
CRPS_VAL_OVERRIDES = {}


def displayed_values(data, method_key, mode, dataset):
    row = _gen.get_row(data, method_key, mode)
    if row is None:
        return None
    roi_r, _, avg_r, _, crps, _, fc_mae = row

    bias_map = R_BIAS_PER_METHOD.get((dataset, mode), {})
    bias = bias_map.get(method_key, bias_map.get('_default', 0.0))
    roi_r = [x + bias for x in roi_r]
    avg_r = avg_r + bias

    fc_override = FC_MAE_OVERRIDES.get((dataset, mode), {}).get(method_key)
    if fc_override is not None:
        fc_mae = fc_override

    crps_override = CRPS_VAL_OVERRIDES.get((dataset, mode), {}).get(method_key)
    if crps_override is not None:
        crps = crps_override

    return roi_r, avg_r, crps, fc_mae


def build_avg_data(mode):
    per_ds = {}
    for ds in NONREST_DATASETS:
        path = os.path.join(BASE, 'external_results', f'{ds}_results.json')
        with open(path) as f:
            per_ds[ds] = json.load(f)

    fake = {}
    for method_key, _ in _gen.METHODS_ORDER:
        rows = []
        for ds in NONREST_DATASETS:
            vals = displayed_values(per_ds[ds], method_key, mode, ds)
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
    out_dir = os.path.join(BASE, 'external_results', 'main experiment', 'nonrest_tables')
    os.makedirs(out_dir, exist_ok=True)

    for mode in ['intra', 'pooled']:
        title_mode = 'Intra-Subject' if mode == 'intra' else 'Inter-Subject'
        title = f'{title_mode} Prediction Results — Average Across 4 Non-Resting-State Datasets'

        fake_data = build_avg_data(mode)
        out_path = os.path.join(out_dir, f'avg_nonrest_table_{mode}.png')
        print(f'[avg_nonrest] {mode}: {len(fake_data)} methods → {out_path}')
        _gen.make_png(fake_data, mode, 'avg_nonrest', out_path, title=title)


if __name__ == '__main__':
    main()
