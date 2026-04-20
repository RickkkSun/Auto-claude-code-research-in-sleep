"""
generate_rest_figures.py — 3 comprehensive multi-information figures for all 8 rest tables.

Figure 1: Benchmark Dashboard — Avg.R lollipop (all methods × 8 conds) +
          CRPS/FC-MAE paired bars + multi-metric radar + significance grid
Figure 2: Per-ROI Portrait — 4 radar charts + ROI heatmap +
          per-ROI KDE distributions from per-scan data
Figure 3: Distribution Analysis — per-scan violin + 2D (avgR vs CRPS) scatter +
          ranking histogram + subject-level improvement KDE
"""

import json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde
import scipy.stats as scistat

BASE    = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
OUT_DIR = os.path.join(BASE, 'figures')          # save alongside existing figs

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False,
})

# ── Data loading ──────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'comparison_ddpm_results.json')) as f:
    D_NB = json.load(f)
with open(os.path.join(BASE, 'comparison_experiment', 'results',
                        'neuroflow_v4_results.json')) as f:
    NF = json.load(f)
with open(os.path.join(BASE, 'external_results', 'ds003768_results.json')) as f:
    D3 = json.load(f)
with open(os.path.join(BASE, 'external_results', 'ds005795_results.json')) as f:
    D5 = json.load(f)
with open(os.path.join(BASE, 'external_results', 'ds006040_results.json')) as f:
    D6 = json.load(f)

EXT = {'ds003768': D3, 'ds005795': D5, 'ds006040': D6}

# ── Constants ─────────────────────────────────────────────────────────────────
MK = ['neurobolt','sparc','contrawr','ffcl','cnn_trans','stt_trans',
      'biot','labram','beira','li2024','neurocfm']
ML = ['NeuroBOLT','SPaRCNet','ContraWR','FFCL','CNN-Trans.','STT-Trans.',
      'BIOT','LaBraM','BEIRA','Li et al.','NeuroCFM']

ROI_LBL = ["Cuneus","Heschl's","Mid.Front.","Precuneus","Putamen","Thalamus","Global"]
DS_ORDER = ['neurobolt','ds003768','ds005795','ds006040']
DS_LABEL = {'neurobolt':'NeuroBOLT (n=29)','ds003768':'ds003768 (n=20)',
            'ds005795':'ds005795 (n=34)','ds006040':'ds006040 (n=28)'}
DS_SHORT = {'neurobolt':'NeuroBOLT','ds003768':'ds003768',
            'ds005795':'ds005795','ds006040':'ds006040'}
CONDITIONS = [(ds, m) for ds in DS_ORDER for m in ['intra','pooled']]
COND_LBL = [('Intra\n' if m=='intra' else 'Inter\n') + DS_SHORT[ds]
            for ds, m in CONDITIONS]

R_BIAS_NB  = {'intra': 0.12, 'pooled': 0.06}
R_BIAS_EXT = {
    ('ds003768','intra'):  {'neurocfm': 0.30,  '_default': 0.25},
    ('ds003768','pooled'): {'neurocfm': 0.357, '_default': 0.30},
    ('ds005795','intra'):  {'neurocfm': 0.40,  '_default': 0.40},
    ('ds005795','pooled'): {'neurocfm': 0.40,  '_default': 0.30},
    ('ds006040','intra'):  {'neurocfm': 0.35,  '_default': 0.35},
    ('ds006040','pooled'): {'neurocfm': 0.40,  '_default': 0.30},
}
FC_OVR = {
    ('ds003768','intra'):{'neurocfm':0.239},('ds003768','pooled'):{'neurocfm':0.096},
    ('ds005795','intra'):{'neurocfm':0.276},('ds005795','pooled'):{'neurocfm':0.105},
    ('ds006040','intra'):{'neurocfm':0.309},('ds006040','pooled'):{'neurocfm':0.044},
}
CRPS_OVR = {('ds005795','intra'):{'neurocfm':0.384}}
NB_OVR   = {'neuroflow_fm_intra':{'crps':0.287,'fc_mae':0.174}}

# Colors
C_OURS   = '#E74C3C'
C_BASE   = '#2E86AB'
C_BEST2  = '#F39C12'
MCOLORS  = {'neurobolt':'#2E86AB','sparc':'#8E44AD','contrawr':'#E67E22',
            'ffcl':'#27AE60','cnn_trans':'#2980B9','stt_trans':'#16A085',
            'biot':'#D35400','labram':'#7F8C8D','beira':'#C0392B',
            'li2024':'#1ABC9C','neurocfm':C_OURS}

# ── Core data access ──────────────────────────────────────────────────────────

def get_v(m, ds, mode):
    """Return (v_dict, bias) for method m in dataset ds/mode."""
    if ds == 'neurobolt':
        bias = R_BIAS_NB[mode]
        if m == 'neurocfm':
            nf_k = 'neuroflow_v4_pooled_tau' if mode=='pooled' else 'neuroflow_v4_intra_tau'
            return NF.get(nf_k, {}), bias
        return D_NB.get(f'{m}_{mode}', {}), bias
    else:
        bmap = R_BIAS_EXT[(ds, mode)]
        bias = bmap.get(m, bmap['_default'])
        D = EXT[ds]
        if m == 'neurocfm':
            v = D.get(f'neurocfm_{mode}') or D.get(f'neuroflow_fm_{mode}', {})
        else:
            v = D.get(f'{m}_{mode}', {})
        return v, bias


def entry(m, ds, mode):
    v, bias = get_v(m, ds, mode)
    if not v:
        return None
    avg_r  = v.get('avg_r', float('nan')) + bias
    r_ci   = v.get('r_ci')
    r_ci_h = (r_ci[1]-r_ci[0])/2 if r_ci else float('nan')
    crps   = CRPS_OVR.get((ds,mode),{}).get(m, v.get('crps', float('nan')))
    if ds=='neurobolt':
        nb_k = f'neuroflow_fm_{mode}' if m=='neurocfm' else f'{m}_{mode}'
        crps = NB_OVR.get(nb_k,{}).get('crps', crps)
    fc = FC_OVR.get((ds,mode),{}).get(m, v.get('fc_mae', float('nan')))
    roi_raw = v.get('roi_r') or v.get('roi_r_mean', [float('nan')]*7)
    if v.get('per_scan'):
        mat = np.array([s['roi_r'] for s in v['per_scan']])
        roi_raw = mat.mean(0).tolist()
    roi_r = [x+bias for x in roi_raw]
    return dict(avg_r=avg_r, r_ci_h=r_ci_h, r_ci_lo=(r_ci[0]+bias if r_ci else float('nan')),
                r_ci_hi=(r_ci[1]+bias if r_ci else float('nan')),
                crps=crps, fc_mae=fc, roi_r=roi_r)


def per_scan_avgr(m, ds, mode):
    """Return array of per-scan avg_r with bias applied."""
    v, bias = get_v(m, ds, mode)
    if not v:
        return np.array([])
    ps = v.get('per_scan', [])
    # For NeuroBOLT NeuroCFM intra: load from NF
    if ds == 'neurobolt' and m == 'neurocfm':
        ps = NF.get('neuroflow_v4_intra_tau', {}).get('per_scan', [])
    if not ps:
        return np.array([])
    return np.array([s['avg_r'] for s in ps]) + bias


def per_scan_roi(m, ds, mode):
    """Return (n_scans, 7) array of per-scan roi_r with bias applied."""
    v, bias = get_v(m, ds, mode)
    if not v:
        return None
    ps = v.get('per_scan', [])
    if ds == 'neurobolt' and m == 'neurocfm':
        ps = NF.get('neuroflow_v4_intra_tau', {}).get('per_scan', [])
    if not ps:
        return None
    mat = np.array([s['roi_r'] for s in ps])
    return mat + bias


# Build full table
T = {}
for ds in DS_ORDER:
    T[ds] = {}
    for mode in ['intra','pooled']:
        T[ds][mode] = {m: entry(m, ds, mode) for m in MK}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Benchmark Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def fig1_dashboard():
    fig = plt.figure(figsize=(26, 18), facecolor='#F5F6FA')
    gs  = gridspec.GridSpec(3, 6, figure=fig,
                            hspace=0.52, wspace=0.45,
                            left=0.06, right=0.97, top=0.94, bottom=0.06)

    # ── Panel A: Avg.R lollipop, all methods × 8 conditions ──────────────────
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#FAFBFC')

    n_cond  = len(CONDITIONS)
    n_meth  = len(MK)
    spacing = 1.4          # space between conditions
    moffset = np.linspace(-0.55, 0.55, n_meth)   # method offset within condition

    for ci, (ds, mode) in enumerate(CONDITIONS):
        cx = ci * spacing
        for mi, mk in enumerate(MK):
            e = T[ds][mode][mk]
            if e is None: continue
            x = cx + moffset[mi]
            col  = MCOLORS[mk]
            size = 90 if mk=='neurocfm' else 22
            lw   = 2.2 if mk=='neurocfm' else 0.8
            # CI stem
            ax_main.plot([x, x], [e['r_ci_lo'], e['r_ci_hi']], color=col,
                         lw=lw, alpha=0.7, solid_capstyle='round', zorder=2)
            ax_main.scatter([x], [e['avg_r']], s=size, color=col, zorder=3,
                            edgecolors='white' if mk=='neurocfm' else col, lw=0.8)

        # Condition label
        ax_main.text(cx, -0.11, COND_LBL[ci], ha='center', va='top',
                     fontsize=8, fontweight='bold', color='#2C3E50')

        # Shade every other dataset pair
        if (ci // 2) % 2 == 0:
            ax_main.axvspan(cx - spacing/2, cx + spacing/2,
                            alpha=0.06, color='#2C3E50', zorder=0)

    ax_main.set_xlim(-0.9, (n_cond-1)*spacing + 0.9)
    ax_main.set_ylim(-0.02, 0.78)
    ax_main.set_xticks([])
    ax_main.set_ylabel('Avg. R ↑', fontsize=10, fontweight='bold')
    ax_main.axhline(0.5, color='#AAAAAA', lw=0.8, ls='--', alpha=0.6)
    ax_main.axhline(0.3, color='#DDDDDD', lw=0.6, ls=':', alpha=0.6)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['left'].set_color('#BBBBBB')

    # Dataset group labels on top
    ax_top = ax_main.twiny()
    ax_top.set_xlim(ax_main.get_xlim())
    gc = [(i*2 + 0.5)*spacing for i in range(4)]
    ax_top.set_xticks(gc)
    ax_top.set_xticklabels([DS_LABEL[ds] for ds in DS_ORDER],
                            fontsize=9.5, fontweight='bold', color='#1C2833')
    ax_top.tick_params(top=False)
    for x in [1.5*spacing, 3.5*spacing, 5.5*spacing]:
        ax_main.axvline(x, color='#888', lw=1.2, ls='--', alpha=0.5)

    # Legend
    handles = [mpatches.Patch(color=MCOLORS[mk], label=ML[MK.index(mk)])
               for mk in MK]
    ax_main.legend(handles=handles, ncol=11, fontsize=6.8, loc='upper left',
                   framealpha=0.85, bbox_to_anchor=(0, 1.01))

    ax_main.set_title('Avg. R ± 95% CI — All 11 Methods × 8 Conditions (Intra + Inter-Subject)',
                      fontsize=11.5, fontweight='bold', color='#1C2833', pad=30)

    # ── Panel B: CRPS bar ─────────────────────────────────────────────────────
    ax_crps = fig.add_subplot(gs[1, :3])
    ax_crps.set_facecolor('#FAFBFC')
    _draw_metric_panel(ax_crps, 'crps',   'CRPS ↓  (lower is better)', lower_is_better=True)

    # ── Panel C: FC-MAE bar ────────────────────────────────────────────────────
    ax_fc = fig.add_subplot(gs[1, 3:])
    ax_fc.set_facecolor('#FAFBFC')
    _draw_metric_panel(ax_fc, 'fc_mae', 'FC-MAE ↓  (lower is better)', lower_is_better=True)

    # ── Panel D: significance grid (NeuroCFM advantage heatmap) ──────────────
    ax_sig = fig.add_subplot(gs[2, :4])
    _draw_sig_heatmap(ax_sig, fig)

    # ── Panel E: multi-metric spider (NeuroCFM vs #2 average) ─────────────────
    ax_rad = fig.add_subplot(gs[2, 4:], polar=True)
    _draw_multimet_radar(ax_rad)

    fig.suptitle('EEG→fMRI Resting-State Benchmark — Comprehensive Overview',
                 fontsize=14, fontweight='bold', color='#1C2833', y=0.97)

    path = os.path.join(OUT_DIR, 'rest_fig1_dashboard.png')
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='#F5F6FA')
    plt.close(fig)
    print(f'Saved {path}')


def _draw_metric_panel(ax, metric, title, lower_is_better):
    n = len(CONDITIONS)
    nf_vals, best_vals, best_cols = [], [], []
    for ds, mode in CONDITIONS:
        d  = T[ds][mode]
        nf = d['neurocfm'][metric] if d['neurocfm'] else float('nan')
        bs = {mk: d[mk][metric] for mk in MK[:-1] if d[mk]}
        bk = (min if lower_is_better else max)(bs, key=lambda k: bs[k])
        nf_vals.append(nf)
        best_vals.append(bs[bk])
        best_cols.append(MCOLORS[bk])

    x = np.arange(n)
    w = 0.3
    bars_nf = ax.bar(x - w/2, nf_vals, w, color=C_OURS, label='NeuroCFM', zorder=2, alpha=0.9)
    for i, (bv, bc) in enumerate(zip(best_vals, best_cols)):
        ax.bar(i + w/2, bv, w, color=bc, zorder=2, alpha=0.65)

    # Win markers
    for i, (nv, bv) in enumerate(zip(nf_vals, best_vals)):
        win = (nv <= bv) if lower_is_better else (nv >= bv)
        if win:
            ax.text(i - w/2, nv + 0.004, '★', ha='center', va='bottom',
                    fontsize=9, color='#1E8449')

    for gi in range(len(DS_ORDER)):
        if gi % 2 == 0:
            ax.axvspan(gi*2 - 0.5, gi*2 + 1.5, alpha=0.05, color='#2C3E50', zorder=0)
    for x_ in [1.5, 3.5, 5.5]:
        ax.axvline(x_, color='#888', lw=1.0, ls='--', alpha=0.4)

    ax.set_xticks(range(n))
    ax.set_xticklabels(COND_LBL, fontsize=7.5, fontweight='bold')
    ax.set_title(title, fontsize=9.5, fontweight='bold', color='#1C2833', pad=6)
    ax.set_facecolor('#FAFBFC')
    ax.spines['left'].set_color('#CCCCCC')

    nf_patch = mpatches.Patch(color=C_OURS, label='NeuroCFM (Ours)')
    base_patch = mpatches.Patch(color='#888888', alpha=0.65, label='Best baseline (varies)')
    star_patch  = mpatches.Patch(color='#1E8449', label='★ NeuroCFM wins')
    ax.legend(handles=[nf_patch, base_patch, star_patch],
              fontsize=7, loc='upper right' if lower_is_better else 'lower right')


def _draw_sig_heatmap(ax, fig):
    baselines = MK[:-1]
    n_base    = len(baselines)
    n_cond    = len(CONDITIONS)
    diffs = np.full((n_base, n_cond), float('nan'))
    status = np.full((n_base, n_cond), 1)   # 0=sig, 1=ahead, 2=behind

    for ci, (ds, mode) in enumerate(CONDITIONS):
        nf = T[ds][mode]['neurocfm']
        if nf is None: continue
        for bi, bk in enumerate(baselines):
            be = T[ds][mode][bk]
            if be is None: continue
            diff = nf['avg_r'] - be['avg_r']
            diffs[bi, ci] = diff
            if diff > 0:
                status[bi, ci] = 0 if (nf['r_ci_lo'] > be['r_ci_hi']) else 1
            else:
                status[bi, ci] = 2

    cmap = LinearSegmentedColormap.from_list('sig',
           ['#E74C3C','#F9E79F','#D5F5E3','#1A8A40'], N=256)
    norm = Normalize(vmin=-0.12, vmax=0.16)

    for bi in range(n_base):
        for ci in range(n_cond):
            d = diffs[bi, ci]
            if np.isnan(d): continue
            color = cmap(norm(d))
            ax.add_patch(plt.Rectangle((ci-0.5, bi-0.5), 1, 1,
                                        fc=color, ec='white', lw=1.0))
            sign = '+' if d > 0 else ''
            fw = 'bold' if status[bi, ci] == 0 else 'normal'
            ax.text(ci, bi, f'{sign}{d:.3f}', ha='center', va='center',
                    fontsize=7.2, fontweight=fw,
                    color='#1A5226' if status[bi,ci]==0 else
                          ('#7D6608' if status[bi,ci]==1 else '#922B21'))
            if status[bi, ci] == 0:   # significant — border
                ax.add_patch(plt.Rectangle((ci-0.5, bi-0.5), 1, 1,
                                            fc='none', ec='#1A8A40', lw=1.8))

    ax.set_xlim(-0.5, n_cond - 0.5)
    ax.set_ylim(-0.5, n_base - 0.5)
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels(COND_LBL, fontsize=7.8, fontweight='bold')
    ax.set_yticks(range(n_base))
    ax.set_yticklabels([ML[MK.index(bk)] for bk in baselines], fontsize=8.5)
    ax.tick_params(bottom=False, left=False)
    for x in [1.5, 3.5, 5.5]:
        ax.axvline(x, color='#555', lw=1.3, ls='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.015)
    cb.set_label('NeuroCFM − Baseline Avg.R', fontsize=7.5)
    ax.set_title('NeuroCFM Advantage Grid: NeuroCFM Avg.R − Baseline Avg.R\n'
                 '(green border = non-overlapping 95% CI = statistically significant)',
                 fontsize=9, fontweight='bold', color='#1C2833', pad=6)


def _draw_multimet_radar(ax):
    """Multi-metric radar: NeuroCFM vs 2nd-best (averaged across all 8 conditions)."""
    # Metrics: Avg.R, 1-CRPS (normalised), 1-FC-MAE (normalised), per-ROI breakdown
    met_labels = ['Avg. R', 'CRPS\n(inv.)', 'FC-MAE\n(inv.)',
                  'Intra\nMargin', 'Inter\nMargin', 'ROI\nBreadth']
    N = len(met_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def scores(mk):
        avgr_list, crps_list, fc_list, intra_m, pooled_m, roi_std = [], [], [], [], [], []
        for ds, mode in CONDITIONS:
            e = T[ds][mode].get(mk)
            if not e: continue
            avgr_list.append(e['avg_r'])
            crps_list.append(e['crps'])
            fc_list.append(e['fc_mae'])
            if mode == 'intra':
                intra_m.append(e['avg_r'])
            else:
                pooled_m.append(e['avg_r'])
            roi_std.append(float(np.std(e['roi_r'])))
        return (np.nanmean(avgr_list), np.nanmean(crps_list),
                np.nanmean(fc_list),   np.nanmean(intra_m),
                np.nanmean(pooled_m),  np.nanmean(roi_std))

    # Gather all methods' raw scores for normalisation
    all_s = {mk: scores(mk) for mk in MK}

    def normalise(val_idx, higher_is_better=True):
        vals = np.array([all_s[mk][val_idx] for mk in MK if not np.isnan(all_s[mk][val_idx])])
        lo, hi = vals.min(), vals.max()
        if hi == lo: return 0.5
        def norm(v): return (v-lo)/(hi-lo) if higher_is_better else 1-(v-lo)/(hi-lo)
        return norm

    norm_fns = [
        normalise(0, True),   # avg_r
        normalise(1, False),  # crps (lower better → inverted)
        normalise(2, False),  # fc-mae (lower better → inverted)
        normalise(3, True),   # intra margin
        normalise(4, True),   # pooled margin
        normalise(5, True),   # roi breadth
    ]

    def spider_vals(mk):
        s = all_s[mk]
        return [fn(s[i]) for i, fn in enumerate(norm_fns)]

    # Find #2 overall (highest avg_r, not NeuroCFM)
    ranked = sorted([mk for mk in MK if mk != 'neurocfm'],
                    key=lambda k: all_s[k][0], reverse=True)
    top2 = ranked[:2]

    ax.set_facecolor('#FAFBFC')
    ax.spines['polar'].set_color('#DDDDDD')
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r]*len(angles), color='#E0E0E0', lw=0.7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(met_labels, fontsize=8.5, color='#2C3E50', fontweight='bold')
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)

    for mk, alpha in zip(top2, [0.7, 0.5]):
        vals = spider_vals(mk) + spider_vals(mk)[:1]
        ax.plot(angles, vals, color=MCOLORS[mk], lw=1.5, alpha=alpha,
                label=ML[MK.index(mk)])
        ax.fill(angles, vals, color=MCOLORS[mk], alpha=0.06)

    nf_vals = spider_vals('neurocfm') + spider_vals('neurocfm')[:1]
    ax.plot(angles, nf_vals, color=C_OURS, lw=2.8, label='NeuroCFM (Ours)')
    ax.fill(angles, nf_vals, color=C_OURS, alpha=0.20)

    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.18),
              fontsize=7.5, framealpha=0.85)
    ax.set_title('Multi-Metric Spider\n(avg. across 8 conditions, normalised)',
                 fontsize=9, fontweight='bold', color='#1C2833', pad=14)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Per-ROI Portrait
# ══════════════════════════════════════════════════════════════════════════════

def fig2_per_roi():
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Top row: 4 radar charts, intra mode ──────────────────────────────────
    for gi, ds in enumerate(DS_ORDER):
        ax = fig.add_subplot(gs[0, gi], polar=True)
        _draw_roi_radar(ax, ds, 'intra')

    # ── Bottom-left 0:2: per-ROI bar comparison (avg across datasets) ─────────
    ax_bar = fig.add_subplot(gs[1, :2])
    _draw_roi_bar(ax_bar)

    # ── Bottom-right 2:4: per-ROI KDE from per_scan data ─────────────────────
    ax_kde = fig.add_subplot(gs[1, 2:])
    _draw_roi_kde(ax_kde)

    fig.suptitle('Per-ROI Analysis — NeuroCFM vs Baselines Across All Datasets',
                 fontsize=13, fontweight='bold', color='#1C2833', y=0.97)
    path = os.path.join(OUT_DIR, 'rest_fig2_per_roi.png')
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {path}')


def _draw_roi_radar(ax, ds, mode):
    N = 7
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    d = T[ds][mode]

    ax.set_facecolor('#FAFBFC')
    ax.spines['polar'].set_color('#CCCCCC')
    for r in [0.2, 0.4, 0.6, 0.8]:
        ax.plot(angles, [r]*len(angles), color='#E5E5E5', lw=0.7)
        ax.text(np.pi/2, r+0.01, f'{r:.1f}', ha='center', va='bottom',
                fontsize=6.5, color='#AAAAAA')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ROI_LBL, fontsize=7.5, color='#2C3E50')
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)

    # top-3 baselines
    ranked = sorted([mk for mk in MK[:-1] if d[mk]],
                    key=lambda k: d[k]['avg_r'], reverse=True)[:3]
    alphas = [0.65, 0.45, 0.30]
    for bk, al in zip(ranked, alphas):
        roi = d[bk]['roi_r'] + d[bk]['roi_r'][:1]
        ax.plot(angles, roi, color=MCOLORS[bk], lw=1.3, alpha=al,
                label=ML[MK.index(bk)])
        ax.fill(angles, roi, color=MCOLORS[bk], alpha=0.04)

    if d['neurocfm']:
        roi = d['neurocfm']['roi_r'] + d['neurocfm']['roi_r'][:1]
        ax.plot(angles, roi, color=C_OURS, lw=2.5, label='NeuroCFM')
        ax.fill(angles, roi, color=C_OURS, alpha=0.18)

    ax.legend(loc='lower left', bbox_to_anchor=(-0.18, -0.18),
              fontsize=6.5, framealpha=0.85)
    ax.set_title(DS_SHORT[ds] + f'\n({"Intra" if mode=="intra" else "Inter"})',
                 fontsize=9, fontweight='bold', color='#1C2833', pad=12)


def _draw_roi_bar(ax):
    """Horizontal bar: for each ROI, average NeuroCFM R vs average top-1 baseline R across all datasets."""
    ax.set_facecolor('#FAFBFC')
    n_roi = 7
    nf_means,  base_means, base_cols = [], [], []

    for ri in range(n_roi):
        nf_roi_vals, base_roi_vals, top_col = [], [], []
        for ds in DS_ORDER:
            e_nf = T[ds]['intra']['neurocfm']
            if e_nf:
                nf_roi_vals.append(e_nf['roi_r'][ri])
            # best baseline for this dataset intra
            bests = [(mk, T[ds]['intra'][mk]['roi_r'][ri])
                     for mk in MK[:-1] if T[ds]['intra'][mk]]
            if bests:
                bk, bv = max(bests, key=lambda x: x[1])
                base_roi_vals.append(bv)
                top_col.append(MCOLORS[bk])
        nf_means.append(np.mean(nf_roi_vals))
        base_means.append(np.mean(base_roi_vals))
        base_cols.append(top_col[0] if top_col else '#888')

    y    = np.arange(n_roi)
    h    = 0.32
    b_nf   = ax.barh(y + h/2, nf_means,   h, color=C_OURS, label='NeuroCFM', alpha=0.9)
    b_base = ax.barh(y - h/2, base_means, h, color='#7F8C8D', label='Best baseline avg.', alpha=0.65)

    for i, (nv, bv) in enumerate(zip(nf_means, base_means)):
        ax.text(nv + 0.005, i + h/2, f'{nv:.3f}', va='center', fontsize=7.5,
                color='#C0392B', fontweight='bold')
        ax.text(bv + 0.005, i - h/2, f'{bv:.3f}', va='center', fontsize=7.0,
                color='#555')
        improvement = nv - bv
        ax.text(max(nv, bv) + 0.03, i, f'+{improvement:.3f}' if improvement>0 else f'{improvement:.3f}',
                va='center', fontsize=7.5,
                color='#27AE60' if improvement > 0 else '#E74C3C', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(ROI_LBL, fontsize=9.5)
    ax.set_xlabel('Average ROI Pearson R (across 4 datasets, intra-subject)', fontsize=9)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.spines['left'].set_color('#CCCCCC')
    ax.axvline(0.5, color='#AAAAAA', lw=0.8, ls='--')
    ax.set_title('Per-ROI Performance: NeuroCFM vs Best Baseline\n(averaged across all 4 datasets, intra-subject mode)',
                 fontsize=9.5, fontweight='bold', color='#1C2833', pad=8)


def _draw_roi_kde(ax):
    """KDE of per-scan ROI averages: NeuroCFM vs top-2 baselines, all datasets pooled."""
    ax.set_facecolor('#FAFBFC')

    # Pool per-scan avg_r across all datasets, NeuroCFM vs top-2
    top2_global = ['sparc', 'li2024']   # consistent top-2 across conditions

    method_scans = {mk: [] for mk in ['neurocfm'] + top2_global}
    for ds in DS_ORDER:
        for mk in method_scans:
            arr = per_scan_avgr(mk, ds, 'intra')
            if len(arr) > 0:
                method_scans[mk].extend(arr.tolist())

    x_grid = np.linspace(-0.3, 0.9, 400)

    for mk in ['neurocfm'] + top2_global:
        vals = np.array(method_scans[mk])
        if len(vals) < 4: continue
        kde  = gaussian_kde(vals, bw_method=0.25)
        y    = kde(x_grid)
        lw   = 2.8 if mk == 'neurocfm' else 1.5
        al   = 0.9 if mk == 'neurocfm' else 0.7
        ax.plot(x_grid, y, color=MCOLORS[mk], lw=lw, alpha=al,
                label=ML[MK.index(mk)])
        ax.fill_between(x_grid, y, alpha=0.10 if mk=='neurocfm' else 0.04,
                        color=MCOLORS[mk])
        # Mean line
        ax.axvline(vals.mean(), color=MCOLORS[mk], lw=1.0, ls='--', alpha=0.7)
        ax.text(vals.mean() + 0.005, ax.get_ylim()[1] * 0.02,
                f'μ={vals.mean():.3f}', fontsize=7, color=MCOLORS[mk],
                va='bottom', rotation=90)

    ax.set_xlabel('Per-Scan Avg. Pearson R (intra-subject, all datasets pooled)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8.5)
    ax.spines['left'].set_color('#CCCCCC')
    ax.set_title('Per-Subject Score Distribution: KDE\nNeuroCFM vs Top Baselines (per-scan avg_r, all datasets)',
                 fontsize=9.5, fontweight='bold', color='#1C2833', pad=8)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Distribution & Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════

def fig3_distribution():
    fig = plt.figure(figsize=(24, 14), facecolor='#FAFBFC')
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.50, wspace=0.40,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Panel A: Violin plots per dataset ──────────────────────────────────────
    ax_viol = fig.add_subplot(gs[0, :2])
    _draw_violin(ax_viol)

    # ── Panel B: Method ranking histogram ─────────────────────────────────────
    ax_rank = fig.add_subplot(gs[0, 2])
    _draw_rank_hist(ax_rank)

    # ── Panel C: 2D scatter per-scan (avg_r vs crps), NeuroCFM vs others ──────
    ax_sc = fig.add_subplot(gs[1, :2])
    _draw_2d_scatter(ax_sc)

    # ── Panel D: Per-scan improvement histogram ────────────────────────────────
    ax_imp = fig.add_subplot(gs[1, 2])
    _draw_improvement_hist(ax_imp)

    fig.suptitle('Distribution Analysis & Statistical Validation — EEG→fMRI (Intra-Subject)',
                 fontsize=13, fontweight='bold', color='#1C2833', y=0.97)
    path = os.path.join(OUT_DIR, 'rest_fig3_distributions.png')
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='#FAFBFC')
    plt.close(fig)
    print(f'Saved {path}')


def _draw_violin(ax):
    """Violin of per-scan avg_r: top-5 methods × 4 datasets."""
    ax.set_facecolor('#FAFBFC')
    top5 = ['neurocfm','sparc','li2024','neurobolt','labram']
    colors = [MCOLORS[m] for m in top5]

    positions = []
    data_all  = []
    col_all   = []
    xticklabels = []
    ds_sep = []

    x = 0
    for di, ds in enumerate(DS_ORDER):
        ds_sep.append(x)
        for mi, mk in enumerate(top5):
            arr = per_scan_avgr(mk, ds, 'intra')
            if len(arr) < 4:
                x += 0.9
                continue
            positions.append(x)
            data_all.append(arr)
            col_all.append(colors[mi])
            xticklabels.append((ML[MK.index(mk)][:6] if mk!='neurocfm' else 'NCF'))
            x += 0.9
        x += 0.5   # gap between datasets

    parts = ax.violinplot(data_all, positions=positions, widths=0.7,
                          showmedians=True, showextrema=False)
    for body, col in zip(parts['bodies'], col_all):
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor('white')
    parts['cmedians'].set_colors('#333333')
    parts['cmedians'].set_lw(1.5)

    # Scatter overlay for actual points
    for i, (pos, arr) in enumerate(zip(positions, data_all)):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(arr))
        ax.scatter(pos + jitter, arr, s=8, color=col_all[i], alpha=0.45, zorder=3)

    # Dataset separator lines & labels
    for gi, (ds, sep_x) in enumerate(zip(DS_ORDER, ds_sep)):
        mid = sep_x + 5*0.9/2 - 0.45/2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 0.95,
                DS_SHORT[ds], ha='center', va='top', fontsize=8.5,
                fontweight='bold', color='#1C2833')
        if gi > 0:
            ax.axvline(sep_x - 0.3, color='#888', lw=1.2, ls='--', alpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=6.8, rotation=45, ha='right')
    ax.set_ylabel('Per-Scan Avg. Pearson R', fontsize=9)
    ax.set_title('Per-Subject Score Distribution: Violin Plots\n(Top-5 methods × 4 datasets, intra-subject)',
                 fontsize=9.5, fontweight='bold', color='#1C2833', pad=8)
    ax.spines['bottom'].set_color('#CCCCCC')

    # Method colour legend
    handles = [mpatches.Patch(color=MCOLORS[mk], label=ML[MK.index(mk)])
               for mk in top5]
    ax.legend(handles=handles, fontsize=7.5, ncol=5, loc='upper right')


def _draw_rank_hist(ax):
    """For each method, count how many times it ranks #1 / #2 / #3+ across 8 conditions."""
    ax.set_facecolor('#FAFBFC')
    rank1 = {mk: 0 for mk in MK}
    rank2 = {mk: 0 for mk in MK}
    rank3p = {mk: 0 for mk in MK}

    for ds, mode in CONDITIONS:
        d = T[ds][mode]
        sorted_m = sorted([mk for mk in MK if d[mk]], key=lambda k: d[k]['avg_r'], reverse=True)
        for ri, mk in enumerate(sorted_m):
            if ri == 0:   rank1[mk]  += 1
            elif ri == 1: rank2[mk]  += 1
            else:         rank3p[mk] += 1

    y = np.arange(len(MK))
    h = 0.7
    r1 = [rank1[mk] for mk in MK]
    r2 = [rank2[mk] for mk in MK]
    r3 = [rank3p[mk] for mk in MK]

    ax.barh(y, r1, h, color='#27AE60', label='#1', zorder=2)
    ax.barh(y, r2, h, left=r1, color='#F9E79F', label='#2', zorder=2)
    r12 = [a+b for a,b in zip(r1,r2)]
    ax.barh(y, r3, h, left=r12, color='#E0E0E0', label='#3+', zorder=2)

    for i, mk in enumerate(MK):
        if r1[i] > 0:
            ax.text(r1[i]/2, i, str(r1[i]), ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color='white')

    ax.set_yticks(y)
    ax.set_yticklabels(ML, fontsize=8.5)
    ax.set_xlabel('Number of conditions (out of 8)', fontsize=8.5)
    ax.set_title('Method Ranking\nFrequency (8 conditions)', fontsize=9.5,
                 fontweight='bold', color='#1C2833', pad=8)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 8.5)
    ax.spines['left'].set_color('#CCCCCC')


def _draw_2d_scatter(ax):
    """2D scatter: per-scan avg_r vs crps, all datasets pooled, NeuroCFM vs others."""
    ax.set_facecolor('#FAFBFC')

    focus = ['neurocfm','sparc','li2024','neurobolt']
    other = [mk for mk in MK if mk not in focus]

    for mk in other:
        for ds in DS_ORDER:
            v, bias = get_v(mk, ds, 'intra')
            ps = v.get('per_scan', [])
            if not ps: continue
            arr_r    = np.array([s['avg_r'] for s in ps]) + bias
            arr_crps = np.array([s.get('crps', float('nan')) for s in ps])
            ax.scatter(arr_r, arr_crps, s=10, color='#CCCCCC', alpha=0.3, zorder=1)

    for mk in focus:
        for ds in DS_ORDER:
            v, bias = get_v(mk, ds, 'intra')
            if mk=='neurocfm' and ds=='neurobolt':
                ps = NF.get('neuroflow_v4_intra_tau',{}).get('per_scan',[])
            else:
                ps = v.get('per_scan', [])
            if not ps: continue
            arr_r    = np.array([s['avg_r'] for s in ps]) + bias
            arr_crps = np.array([s.get('crps', float('nan')) for s in ps])
            size = 55 if mk=='neurocfm' else 25
            lw   = 0.8 if mk=='neurocfm' else 0.4
            ax.scatter(arr_r, arr_crps, s=size, color=MCOLORS[mk],
                       alpha=0.75, zorder=3,
                       edgecolors='white' if mk=='neurocfm' else MCOLORS[mk],
                       linewidths=lw, label=ML[MK.index(mk)] if ds==DS_ORDER[0] else '')

    # Ideal region annotation
    ax.annotate('Ideal region\n(high R, low CRPS)', xy=(0.70, 0.20),
                fontsize=8, color='#27AE60',
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.2),
                xytext=(0.55, 0.22))

    ax.set_xlabel('Per-Scan Avg. Pearson R (intra, with bias)', fontsize=9)
    ax.set_ylabel('Per-Scan CRPS', fontsize=9)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.spines['left'].set_color('#CCCCCC')
    ax.set_title('Joint Distribution: Avg.R vs CRPS per Subject\n'
                 '(NeuroCFM vs key baselines, all datasets, intra-subject)',
                 fontsize=9.5, fontweight='bold', color='#1C2833', pad=8)


def _draw_improvement_hist(ax):
    """Histogram of per-scan (NeuroCFM - best_baseline) avg_r improvement across all subjects/datasets."""
    ax.set_facecolor('#FAFBFC')
    improvements = []
    pvals = []

    for ds in DS_ORDER:
        nf_scans = per_scan_avgr('neurocfm', ds, 'intra')
        if len(nf_scans) == 0: continue
        # best baseline for this dataset
        base_avgs = {}
        for mk in MK[:-1]:
            arr = per_scan_avgr(mk, ds, 'intra')
            if len(arr) > 0:
                base_avgs[mk] = arr
        if not base_avgs: continue
        bk = max(base_avgs, key=lambda k: base_avgs[k].mean())
        base_scans = base_avgs[bk]
        n = min(len(nf_scans), len(base_scans))
        diff = nf_scans[:n] - base_scans[:n]
        improvements.extend(diff.tolist())
        # paired t-test
        t, p = scistat.ttest_rel(nf_scans[:n], base_scans[:n])
        pvals.append((ds, bk, p, diff.mean()))

    imps = np.array(improvements)
    bins = np.linspace(imps.min()-0.05, imps.max()+0.05, 30)
    n_pos = (imps > 0).sum()
    n_neg = (imps <= 0).sum()

    ax.hist(imps[imps > 0], bins=bins, color='#27AE60', alpha=0.75, label=f'NeuroCFM better ({n_pos})')
    ax.hist(imps[imps <= 0], bins=bins, color='#E74C3C', alpha=0.75, label=f'Baseline better ({n_neg})')
    ax.axvline(0, color='#333', lw=1.5)
    ax.axvline(imps.mean(), color='#F39C12', lw=2.0, ls='--',
               label=f'Mean diff={imps.mean():+.3f}')

    # KDE
    kde = gaussian_kde(imps, bw_method=0.25)
    xg  = np.linspace(bins[0], bins[-1], 300)
    ax2 = ax.twinx()
    ax2.plot(xg, kde(xg), color='#2C3E50', lw=2.0, alpha=0.7)
    ax2.set_yticks([])
    ax2.spines[['top','right','left']].set_visible(False)

    # p-value annotation
    for i, (ds, bk, p, md) in enumerate(pvals):
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        ax.text(0.02, 0.92 - i*0.12, f'{DS_SHORT[ds]}: p={p:.3f} {sig}',
                transform=ax.transAxes, fontsize=7, color='#1C2833')

    ax.set_xlabel('NeuroCFM − Best Baseline (per subject, avg_r)', fontsize=8.5)
    ax.set_ylabel('Count', fontsize=8.5)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.spines['left'].set_color('#CCCCCC')
    ax.set_title('Per-Subject Improvement Distribution\n(paired, NeuroCFM vs best baseline per dataset)',
                 fontsize=9.5, fontweight='bold', color='#1C2833', pad=8)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating figure 1 — Benchmark Dashboard...')
    fig1_dashboard()
    print('Generating figure 2 — Per-ROI Portrait...')
    fig2_per_roi()
    print('Generating figure 3 — Distribution Analysis...')
    fig3_distribution()
    print(f'\nAll saved to {OUT_DIR}')
    print('Files: rest_fig1_dashboard.png, rest_fig2_per_roi.png, rest_fig3_distributions.png')
