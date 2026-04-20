"""
train_neurocfm_external.py — NeuroCFM (NeuroFlow v4) intra-subject evaluation
on external preprocessed datasets.

Usage:
  python train_neurocfm_external.py --dataset ds003768
  python train_neurocfm_external.py --dataset ds005795 --out external_results/ds005795_results.json
"""

import sys, os, gc, json, math, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import mne
mne.set_log_level('WARNING')
from scipy.signal import butter, filtfilt

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
sys.path.insert(0, os.path.join(BASE, 'code'))
sys.path.insert(0, BASE)

# ── Load NeuroCFM internals from the original script ─────────────────────────
import comparison_experiment.scripts.train_neuroflow_v4_intra as _nfm

DEVICE         = _nfm.DEVICE
ROI_COLS       = _nfm.ROI_COLS
ROIS           = _nfm.ROIS
CH_NAMES       = _nfm.CH_NAMES
TMIN           = _nfm.TMIN
CROP           = _nfm.CROP
EVENT          = _nfm.EVENT   # 'R149'

load_all_backbones     = _nfm.load_all_backbones
extract_1400d_features = _nfm.extract_1400d_features
train_scan             = _nfm.train_scan

# ── External dataset configs ──────────────────────────────────────────────────
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

# ── Data loading for external datasets ───────────────────────────────────────
from dataset_maker import preproc


def discover_scans(data_root: str):
    eeg_dir = os.path.join(data_root, 'EEG')
    return sorted(f.replace('_eeg.set', '')
                  for f in os.listdir(eeg_dir) if f.endswith('_eeg.set'))


def load_scan_ext(patient_name: str, data_root: str, tr: float):
    eeg_path = os.path.join(data_root, 'EEG', f'{patient_name}_eeg.set')
    fm_path  = os.path.join(data_root, 'fMRI_difumo64', f'{patient_name}_difumo64_roi.pkl')

    if not os.path.exists(eeg_path) or not os.path.exists(fm_path):
        return None

    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
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
        fm, _ = preproc.normalize_data(fm)
        try:
            ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        except KeyError:
            del raw
            return None
        n = len(ep['eeg'])
        if eeg_all is None:
            eeg_all = torch.stack(
                [torch.tensor(x, dtype=torch.float32) for x in ep['eeg']])
        all_fmri.append(torch.tensor(
            [float(np.asarray(x).flat[0]) for x in ep['fmri']], dtype=torch.float32))

    del raw
    fmri_all = torch.stack(all_fmri, dim=1)

    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / tr)
    return (eeg_all[:valcrop], fmri_all[:valcrop],
            eeg_all[valcrop:], fmri_all[valcrop:])


# ── Main ──────────────────────────────────────────────────────────────────────

def run_intra(ds_key, data_root, tr, out_path):
    """Per-scan intra-subject evaluation."""
    backbones = load_all_backbones()
    scan_results, scan_names = [], []

    for pat in discover_scans(data_root):
        print(f'\n[{pat}]', flush=True)
        try:
            data = load_scan_ext(pat, data_root, tr)
            if data is None:
                print('  SKIP (missing)'); continue
            eeg_tv, fmri_tv, eeg_te, fmri_te = data
            if len(eeg_te) < 5:
                print('  SKIP (short test)'); continue
            feat_tv = extract_1400d_features(backbones, eeg_tv)
            feat_te = extract_1400d_features(backbones, eeg_te)
            res = train_scan(feat_tv, fmri_tv, feat_te, fmri_te, epochs=200, lr=3e-4, bs=64)
            if res is None:
                print('  SKIP (too short)'); continue
            scan_results.append(res); scan_names.append(pat)
            print(f'  R={res["avg_r"]:.4f}  CRPS={res["crps"]:.4f}  '
                  f'FC-MAE={res["fc_mae"]:.4f}  tau={res["tau"]:.3f}')
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f'  SKIP ({e})')

    if not scan_results:
        print('No valid scan results!'); return

    S = len(scan_results)
    all_r    = [r['avg_r']  for r in scan_results]
    all_crps = [r['crps']   for r in scan_results]
    all_fc   = [r['fc_mae'] for r in scan_results]
    roi_r_mat = np.array([r['roi_r'] for r in scan_results])

    rng = np.random.default_rng(42)
    crps_boot = [np.array(all_crps)[rng.integers(0, S, S)].mean() for _ in range(500)]
    r_boot    = [np.array(all_r)[rng.integers(0, S, S)].mean() for _ in range(500)]
    crps_ci   = (float(np.quantile(crps_boot, 0.025)), float(np.quantile(crps_boot, 0.975)))
    r_ci      = (float(np.quantile(r_boot, 0.025)),    float(np.quantile(r_boot, 0.975)))

    summary = {
        'avg_r': float(np.nanmean(all_r)), 'avg_r_std': float(np.nanstd(all_r)),
        'r_ci': r_ci, 'roi_r': np.nanmean(roi_r_mat, axis=0).tolist(),
        'roi_r_mean': np.nanmean(roi_r_mat, axis=0).tolist(),
        'crps': float(np.mean(all_crps)), 'crps_ci': crps_ci,
        'fc_mae': float(np.mean(all_fc)), 'mode': 'intra', 'dataset': ds_key,
        'per_scan': [{'scan': n, **r} for n, r in zip(scan_names, scan_results)],
    }
    print(f'\n=== INTRA SUMMARY ({S} scans, {ds_key}) ===')
    print(f'avg_R={summary["avg_r"]:.4f}±{summary["avg_r_std"]:.4f}  '
          f'R_95CI=[{r_ci[0]:.4f},{r_ci[1]:.4f}]')
    print(f'CRPS={summary["crps"]:.4f} [{crps_ci[0]:.4f},{crps_ci[1]:.4f}]  '
          f'FC-MAE={summary["fc_mae"]:.4f}')

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f: existing = json.load(f)
    existing['neurocfm_intra'] = summary
    existing['neuroflow_fm_intra'] = summary
    with open(out_path, 'w') as f: json.dump(existing, f, indent=2)
    print(f'Results saved → {out_path}')


def run_pooled(ds_key, data_root, tr, out_path):
    """Pooled (inter-subject): one FM head trained on all subjects, tested on held-out."""
    from comparison_experiment.scripts.train_neuroflow_v4_intra import (
        ResidualFMHead, ot_cfm_loss, crps_score, corr_mat,
        N_ROIS, FEAT_TOTAL
    )
    import torch.nn as nn, torch.nn.functional as F
    from scipy.stats import pearsonr

    backbones = load_all_backbones()

    all_feat_tv, all_fmri_tv = [], []
    all_feat_te, all_fmri_te = [], []
    scan_names = []

    for pat in discover_scans(data_root):
        data = load_scan_ext(pat, data_root, tr)
        if data is None: continue
        eeg_tv, fmri_tv, eeg_te, fmri_te = data
        if len(eeg_te) < 3: continue
        feat_tv = extract_1400d_features(backbones, eeg_tv)
        feat_te = extract_1400d_features(backbones, eeg_te)
        all_feat_tv.append(feat_tv); all_fmri_tv.append(fmri_tv)
        all_feat_te.append(feat_te); all_fmri_te.append(fmri_te)
        scan_names.append(pat)
        print(f'  [{pat}] tv={len(feat_tv)} te={len(feat_te)}', flush=True)
        gc.collect(); torch.cuda.empty_cache()

    if not all_feat_tv:
        print('No valid scans!'); return

    feat_tv_all = torch.cat(all_feat_tv)
    fmri_tv_all = torch.cat(all_fmri_tv)
    feat_te_all = torch.cat(all_feat_te)
    fmri_te_all = torch.cat(all_fmri_te)

    n_tv  = len(feat_tv_all)
    split = int(0.8 / 0.9 * n_tv)
    feat_tr, fmri_tr   = feat_tv_all[:split], fmri_tv_all[:split]
    feat_val, fmri_val = feat_tv_all[split:], fmri_tv_all[split:]
    print(f'\nPooled: train={len(feat_tr)}, val={len(feat_val)}, test={len(feat_te_all)}')

    # Phase 1: Linear mean
    lin = nn.Linear(FEAT_TOTAL, N_ROIS).to(DEVICE)
    opt_lin = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 128
    for ep in range(50):
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            loss = F.mse_loss(lin(feat_tr[idx].to(DEVICE)), fmri_tr[idx].to(DEVICE))
            opt_lin.zero_grad(); loss.backward(); opt_lin.step()
    lin.eval()

    with torch.no_grad():
        mu_tr  = torch.cat([lin(feat_tr[i:i+512].to(DEVICE)).cpu() for i in range(0,len(feat_tr),512)])
        mu_val = torch.cat([lin(feat_val[i:i+512].to(DEVICE)).cpu() for i in range(0,len(feat_val),512)])
    res_tr  = fmri_tr  - mu_tr
    res_val = fmri_val - mu_val
    sig = res_tr.std(0).clamp(min=1e-8)
    res_tr_n, res_val_n = res_tr / sig, res_val / sig

    # Phase 2: FM on residuals
    head = ResidualFMHead(feat_dim=FEAT_TOTAL).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 400
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=epochs * max(1, math.ceil(len(feat_tr)/bs)))
    best_crps, best_state, no_imp = 1e9, None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(feat_tr))
        for i in range(0, len(feat_tr), bs):
            idx = perm[i:i+bs]
            loss = ot_cfm_loss(head, feat_tr[idx].to(DEVICE), res_tr_n[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1) % 20 == 0:
            head.eval()
            with torch.no_grad():
                sv = head.sample(feat_val.to(DEVICE), n_samples=10, num_steps=50)
                sv = (sv * sig.to(DEVICE)).cpu()
            vc = crps_score(sv, res_val, exact=False)
            if vc < best_crps:
                best_crps = vc; best_state = {k: v.cpu().clone() for k,v in head.state_dict().items()}; no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 10: break
            if (ep+1) % 100 == 0: print(f'  ep={ep+1} val_crps={vc:.4f}', flush=True)
    if best_state: head.load_state_dict(best_state)
    head.eval()

    # Tau optimization
    with torch.no_grad():
        sv = head.sample(feat_val.to(DEVICE), n_samples=50, num_steps=100)
        sv = (sv * sig.to(DEVICE)).cpu()
    best_tau_crps, best_tau = 1e9, 1.0
    mu_sv = sv.mean(0)
    for tau in torch.linspace(0.05, 1.5, 60):
        sc = mu_sv.unsqueeze(0) + tau * (sv - mu_sv.unsqueeze(0))
        c  = crps_score(sc, res_val, exact=False)
        if c < best_tau_crps: best_tau_crps = c; best_tau = float(tau)

    # Test evaluation
    with torch.no_grad():
        st = head.sample(feat_te_all.to(DEVICE), n_samples=100, num_steps=100)
        st = (st * sig.to(DEVICE)).cpu()
        mu_te = torch.cat([lin(feat_te_all[i:i+512].to(DEVICE)).cpu() for i in range(0,len(feat_te_all),512)])
    mu_st = st.mean(0)
    st_scaled = mu_st.unsqueeze(0) + best_tau * (st - mu_st.unsqueeze(0))
    st_total  = mu_te.unsqueeze(0) + st_scaled

    pred_mean = st_total.mean(0)
    roi_r = [float(pearsonr(pred_mean[:,j].numpy(), fmri_te_all[:,j].numpy())[0]) for j in range(N_ROIS)]
    crps_te = crps_score(st_total, fmri_te_all, exact=True)
    fc_mae  = (corr_mat(pred_mean) - corr_mat(fmri_te_all)).abs().mean().item()

    rng = np.random.default_rng(42)
    S = len(scan_names)
    # Per-scan test metrics for CI
    te_sizes = [len(x) for x in all_fmri_te]
    te_offsets = [sum(te_sizes[:i]) for i in range(S)]
    per_scan_r = []
    for i in range(S):
        a, b = te_offsets[i], te_offsets[i]+te_sizes[i]
        pm_i = pred_mean[a:b]; tgt_i = fmri_te_all[a:b]
        roi_r_i = [float(pearsonr(pm_i[:,j].numpy(), tgt_i[:,j].numpy())[0]) for j in range(N_ROIS)]
        per_scan_r.append(float(np.nanmean(roi_r_i)))
    r_boot = [np.array(per_scan_r)[rng.integers(0,S,S)].mean() for _ in range(500)]
    r_ci   = (float(np.quantile(r_boot,0.025)), float(np.quantile(r_boot,0.975)))

    summary = {
        'avg_r': float(np.nanmean(roi_r)), 'roi_r': roi_r, 'roi_r_mean': roi_r,
        'r_ci': r_ci, 'crps': crps_te, 'fc_mae': fc_mae,
        'tau': best_tau, 'mode': 'pooled', 'dataset': ds_key,
        'n_scans': S, 'n_test': len(fmri_te_all),
    }
    print(f'\n=== POOLED SUMMARY ({S} scans, {ds_key}) ===')
    print(f'avg_R={summary["avg_r"]:.4f}  R_95CI=[{r_ci[0]:.4f},{r_ci[1]:.4f}]')
    print(f'CRPS={crps_te:.4f}  FC-MAE={fc_mae:.4f}  tau={best_tau:.3f}')

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f: existing = json.load(f)
    existing['neurocfm_pooled'] = summary
    existing['neuroflow_fm_pooled'] = summary
    with open(out_path, 'w') as f: json.dump(existing, f, indent=2)
    print(f'Results saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=list(EXTERNAL_DATASET_CONFIGS.keys()))
    parser.add_argument('--mode', default='intra', choices=['intra', 'pooled', 'both'])
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--out', default=None,
                        help='Output JSON. Default: external_results/{dataset}_results.json')
    args = parser.parse_args()

    ds_key    = args.dataset
    data_root = args.data_root or os.path.join(BASE, 'external_processed', ds_key)
    tr        = EXTERNAL_DATASET_CONFIGS[ds_key]['tr']
    out_dir   = os.path.join(BASE, 'external_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path  = args.out or os.path.join(out_dir, f'{ds_key}_results.json')

    print(f'NeuroCFM external | dataset={ds_key} mode={args.mode} TR={tr}')
    print(f'Device: {DEVICE}')

    modes = ['intra', 'pooled'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        # Skip if already done
        if os.path.exists(out_path):
            with open(out_path) as f: existing = json.load(f)
            key = f'neurocfm_{mode}'
            if key in existing:
                print(f'[{mode}] Already done — skipping'); continue
        print(f'\n=== NeuroCFM {mode} on {ds_key} ===')
        if mode == 'intra':
            run_intra(ds_key, data_root, tr, out_path)
        else:
            run_pooled(ds_key, data_root, tr, out_path)
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
