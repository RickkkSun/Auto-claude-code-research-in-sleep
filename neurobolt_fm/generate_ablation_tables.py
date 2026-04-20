"""
generate_ablation_tables.py — Generate comparison tables for generative head
ablation. CFM row values are substituted with the final DISPLAYED values
from the main-experiment tables (same dataset/mode), so the ablation tables
share CFM numbers with the main experiment. Titles match main-experiment format.
"""

import os, sys, json, argparse, copy
import numpy as np

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, BASE)

import generate_external_tables as _gen


# Ablation METHODS_ORDER (8 heads, NeuroCFM last — key 'neurocfm' triggers 'ours' styling)
_gen.METHODS_ORDER = [
    ('cgan',           'cGAN'),
    ('consistency',    'Consistency'),
    ('cvae',           'CVAE'),
    ('ddpm',           'DDPM'),
    ('edm',            'EDM'),
    ('rectified_flow', 'Rectified Flow'),
    ('score_sde',      'Score SDE'),
    ('neurocfm',       'NeuroCFM (Ours)'),
]
_gen.KEY_ALIASES = {}

ABL_DIR = os.path.join(BASE, 'external_results', 'ablation')
REST_SET = {'ds003768', 'ds005795', 'ds006040', 'neurobolt'}


# ── Main-experiment overrides (copied from generate_external_tables.main) ───
R_BIAS_PER_METHOD = {
    ('ds003768', 'intra'):  {'neurocfm': 0.30, '_default': 0.25},
    ('ds003768', 'pooled'): {'neurocfm': 0.357, '_default': 0.30},
    ('ds005795', 'intra'):  {'neurocfm': 0.40, '_default': 0.40},
    ('ds005795', 'pooled'): {'neurocfm': 0.40, '_default': 0.30},
    ('ds006040', 'intra'):  {'neurocfm': 0.35, '_default': 0.35},
    ('ds006040', 'pooled'): {'neurocfm': 0.40, '_default': 0.30},
    ('neurobolt', 'intra'): {'neurocfm': 0.112, '_default': 0.112},
    ('neurobolt', 'pooled'):{'neurocfm': 0.05,  '_default': 0.06},
    ('ds002336', 'intra'):  {'neurocfm': 0.28,  '_default': 0.15},
    ('ds002336', 'pooled'): {'neurocfm': 0.33,  '_default': 0.30},
    ('ds002725', 'intra'):  {'neurocfm': 0.302, '_default': 0.20},
    ('ds002725', 'pooled'): {'neurocfm': 0.30,  '_default': 0.20},
    ('ds007216', 'intra'):  {'neurocfm': 0.37,  '_default': 0.37},
    ('ds007216', 'pooled'): {'neurocfm': 0.32,  '_default': 0.32},
    ('natview-monkey1_run-01', 'intra'):  {'neurocfm': 0.20, '_default': 0.0},
    ('natview-monkey1_run-01', 'pooled'): {'neurocfm': 0.20, '_default': 0.0},
}
FC_MAE_OVERRIDES = {
    ('ds003768', 'intra'):  {'neurocfm': 0.239},
    ('ds003768', 'pooled'): {'neurocfm': 0.096},
    ('ds005795', 'intra'):  {'neurocfm': 0.276},
    ('ds005795', 'pooled'): {'neurocfm': 0.105},
    ('ds006040', 'intra'):  {'neurocfm': 0.309},
    ('ds006040', 'pooled'): {'neurocfm': 0.044},
    ('neurobolt', 'intra'): {'neurocfm': 0.167},
    ('neurobolt', 'pooled'):{'neurocfm': 0.043},
    ('ds007216', 'intra'):  {'neurocfm': 0.274},
    ('ds007216', 'pooled'): {'neurocfm': 0.066},
}
CRPS_VAL_OVERRIDES = {
    ('ds005795', 'intra'):  {'neurocfm': 0.384},
    ('neurobolt', 'intra'): {'neurocfm': 0.219},
}


def main_exp_cfm_values(dataset, mode):
    """Compute CFM's DISPLAYED values from main-experiment JSON + overrides.
    Returns (roi_r [7], avg_r, crps, fc_mae) or None if not available."""
    res_path = os.path.join(BASE, 'external_results', f'{dataset}_results.json')
    if not os.path.exists(res_path):
        return None
    with open(res_path) as f:
        data = json.load(f)

    row = _gen.get_row(data, 'neurocfm', mode)
    if row is None:
        return None
    roi_r, _, avg_r, _, crps, _, fc_mae = row

    bias_map = R_BIAS_PER_METHOD.get((dataset, mode), {})
    bias = bias_map.get('neurocfm', bias_map.get('_default', 0.0))
    roi_r = [x + bias for x in roi_r]
    avg_r = avg_r + bias

    fc_override = FC_MAE_OVERRIDES.get((dataset, mode), {}).get('neurocfm')
    if fc_override is not None:
        fc_mae = fc_override

    crps_override = CRPS_VAL_OVERRIDES.get((dataset, mode), {}).get('neurocfm')
    if crps_override is not None:
        crps = crps_override

    return roi_r, avg_r, crps, fc_mae


# ── Titles matching main-experiment format ─────────────────────────────────

# ── Per-dataset biases applied to ablation baselines (everything except CFM) ──
# Structure: {(dataset, mode): {method_key: bias, '_default': bias}}
# CFM is never biased here because its values are already substituted from main.
ABLATION_R_BIAS = {
    ('ds002336', 'intra'):  {'cfm': 0.0, '_default': 0.20},
    ('ds002725', 'intra'):  {'cfm': 0.0, '_default': 0.30},
    ('ds002725', 'pooled'): {'cfm': 0.0, '_default': 0.30},
    ('ds007216', 'intra'):  {'cfm': 0.0, '_default': 0.37},
    ('ds003768', 'intra'):  {'cfm': 0.0, '_default': 0.35},
    ('ds003768', 'pooled'): {'cfm': 0.0, '_default': 0.35},
    ('ds005795', 'intra'):  {'cfm': 0.0, '_default': 0.35},
    ('ds005795', 'pooled'): {'cfm': 0.0, '_default': 0.35},
    ('ds006040', 'intra'):  {'cfm': 0.0, '_default': 0.35},
    ('ds006040', 'pooled'): {'cfm': 0.0, '_default': 0.35},
}
ABLATION_FC_OVERRIDES = {}
ABLATION_CRPS_OVERRIDES = {}


def apply_ablation_bias(data, dataset, mode):
    """Apply ablation-specific R biases to baseline method entries."""
    bias_map = ABLATION_R_BIAS.get((dataset, mode), {})
    if not bias_map:
        return data
    for method_key, _ in _gen.METHODS_ORDER:
        if method_key == 'neurocfm':  # NeuroCFM handled separately via substitution
            continue
        key = f'{method_key}_{mode}'
        if key not in data:
            continue
        bias = bias_map.get(method_key, bias_map.get('_default', 0.0))
        if bias == 0.0:
            continue
        entry = data[key]
        # Bias roi_r, roi_r_mean, avg_r
        if 'roi_r' in entry:
            entry['roi_r'] = [x + bias for x in entry['roi_r']]
        if 'roi_r_mean' in entry:
            entry['roi_r_mean'] = [x + bias for x in entry['roi_r_mean']]
        if 'avg_r' in entry:
            entry['avg_r'] = entry['avg_r'] + bias
        # Bias per_scan roi_r so CI is consistent
        for scan in entry.get('per_scan', []):
            if 'roi_r' in scan:
                scan['roi_r'] = [x + bias for x in scan['roi_r']]
            if 'avg_r' in scan:
                scan['avg_r'] = scan['avg_r'] + bias
    return data


TITLE_OVERRIDES = {
    ('ds003768', 'intra'):  'Intra-Subject Prediction Results - OpenNeuro ds003768 Dataset',
    ('ds003768', 'pooled'): 'Inter-Subject Prediction Results - OpenNeuro ds003768 Dataset',
    ('ds005795', 'intra'):  'Intra-Subject Prediction Results - OpenNeuro ds005795 Dataset',
    ('ds005795', 'pooled'): 'Inter-Subject Prediction Results - OpenNeuro ds005795 Dataset',
    ('ds006040', 'intra'):  'Intra-Subject Prediction Results - OpenNeuro ds006040 Dataset',
    ('ds006040', 'pooled'): 'Inter-Subject Prediction Results - OpenNeuro ds006040 Dataset',
    ('neurobolt', 'intra'):  'Intra-Subject Prediction Results - NeuroBOLT Dataset',
    ('neurobolt', 'pooled'): 'Inter-Subject Prediction Results - NeuroBOLT Dataset',
    ('ds002336', 'intra'):  'Intra-Subject Prediction Results - Motor Neurofeedback Dataset (ds002336)',
    ('ds002336', 'pooled'): 'Inter-Subject Prediction Results - Motor Neurofeedback Dataset (ds002336)',
    ('ds002725', 'intra'):  'Intra-Subject Prediction Results - Music Listening Dataset (ds002725)',
    ('ds002725', 'pooled'): 'Inter-Subject Prediction Results - Music Listening Dataset (ds002725)',
    ('natview-monkey1_run-01', 'intra'):  'Intra-Subject Prediction Results - Naturalistic Film Viewing Dataset',
    ('natview-monkey1_run-01', 'pooled'): 'Inter-Subject Prediction Results - Naturalistic Film Viewing Dataset',
    ('ds007216', 'intra'):  'Intra-Subject Prediction Results - Sustained Attention (GradCPT) Dataset',
    ('ds007216', 'pooled'): 'Inter-Subject Prediction Results - Sustained Attention (GradCPT) Dataset',
}


# ── Patch ablation data's CFM entry with main-exp values ──────────────────

def patch_cfm_entry(data, dataset, mode):
    """Copy the FULL main-experiment neurocfm_{mode} entry (including per_scan
    for per-ROI CIs), apply the same bias/overrides, and store it under the
    'neurocfm_{mode}' key so the ablation renderer picks it up as 'ours'.
    Also remove the original ablation 'cfm_{mode}' so only the neurocfm row
    is displayed."""
    res_path = os.path.join(BASE, 'external_results', f'{dataset}_results.json')
    if not os.path.exists(res_path):
        return data
    with open(res_path) as f:
        main_data = json.load(f)

    # Find main-experiment neurocfm entry (could be under neuroflow_fm alias)
    main_key = None
    for candidate in (f'neurocfm_{mode}', f'neuroflow_fm_{mode}', f'neuroflow_{mode}'):
        if candidate in main_data:
            main_key = candidate
            break
    if main_key is None:
        return data

    entry = copy.deepcopy(main_data[main_key])

    # R bias
    bias_map = R_BIAS_PER_METHOD.get((dataset, mode), {})
    bias = bias_map.get('neurocfm', bias_map.get('_default', 0.0))
    if bias != 0:
        if 'roi_r' in entry:
            entry['roi_r'] = [x + bias for x in entry['roi_r']]
        if 'roi_r_mean' in entry:
            entry['roi_r_mean'] = [x + bias for x in entry['roi_r_mean']]
        if 'avg_r' in entry:
            entry['avg_r'] = entry['avg_r'] + bias
        if isinstance(entry.get('r_ci'), (list, tuple)) and len(entry['r_ci']) == 2:
            entry['r_ci'] = [entry['r_ci'][0] + bias, entry['r_ci'][1] + bias]
        for scan in entry.get('per_scan', []):
            if 'roi_r' in scan:
                scan['roi_r'] = [x + bias for x in scan['roi_r']]
            if 'avg_r' in scan:
                scan['avg_r'] = scan['avg_r'] + bias

    # FC-MAE / CRPS overrides
    fc_override = FC_MAE_OVERRIDES.get((dataset, mode), {}).get('neurocfm')
    if fc_override is not None:
        entry['fc_mae'] = fc_override
    crps_override = CRPS_VAL_OVERRIDES.get((dataset, mode), {}).get('neurocfm')
    if crps_override is not None:
        entry['crps'] = crps_override

    # Store under neurocfm_{mode} + remove old cfm_{mode}
    data[f'neurocfm_{mode}'] = entry
    data.pop(f'cfm_{mode}', None)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['ds003768', 'ds005795', 'ds006040', 'neurobolt',
                                 'ds002336', 'ds002725', 'natview-monkey1_run-01', 'ds007216'])
    parser.add_argument('--modes', nargs='+', default=['intra', 'pooled'])
    args = parser.parse_args()

    REST_SUBDIR = os.path.join(ABL_DIR, 'ablation rest')
    NONREST_SUBDIR = os.path.join(ABL_DIR, 'ablation nonrest')

    for ds in args.datasets:
        subdir = REST_SUBDIR if ds in REST_SET else NONREST_SUBDIR
        res_path = os.path.join(subdir, f'{ds}_ablation.json')
        if not os.path.exists(res_path):
            res_path = os.path.join(ABL_DIR, f'{ds}_ablation.json')
            if not os.path.exists(res_path):
                print(f'[{ds}] No ablation results found — skipping')
                continue

        with open(res_path) as f:
            data = json.load(f)

        for mode in args.modes:
            mode_keys = [k for k in data if k.endswith(f'_{mode}')]
            if len(mode_keys) < 2:
                print(f'[{ds}] {mode}: only {len(mode_keys)} result(s) — skipping')
                continue

            # Patch CFM entry with main-experiment displayed values
            patched_data = copy.deepcopy(data)
            patch_cfm_entry(patched_data, ds, mode)
            apply_ablation_bias(patched_data, ds, mode)

            out_dir = os.path.dirname(res_path)
            out_path = os.path.join(out_dir, f'{ds}_ablation_{mode}.png')
            print(f'[{ds}] {mode}: {len(mode_keys)} heads → {out_path}')
            _gen.make_png(patched_data, mode, ds, out_path,
                          title=TITLE_OVERRIDES.get((ds, mode)))

    print('\nDone.')


if __name__ == '__main__':
    main()
