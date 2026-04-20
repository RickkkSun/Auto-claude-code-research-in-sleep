"""
FM-NeuroBOLT v2+ — FM with partial backbone fine-tuning for higher R.

Key difference from FM v2: unfreeze the last 2 transformer blocks + MSS module
of the LaBraM backbone, using a very small backbone lr (1e-5) while keeping
the FM head lr at 3e-4.

This allows the backbone to specialize its representations for each ROI,
while the FM head learns the flow matching objective.

Expected: Avg.R > 0.446 (Ridge baseline), demonstrating FM+fine-tuning advantage.
"""

import sys, os, gc, json, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import mne
mne.set_log_level('WARNING')
from einops import rearrange
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from timm.models import create_model
import models.model
from dataset_maker import preproc

BASE     = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CKPT_DIR = f'{BASE}/code/checkpoints'
DATA_ROOT= f'{BASE}/data'
FT_DIR   = f'{BASE}/checkpoints_fm_v2ft'
os.makedirs(FT_DIR, exist_ok=True)
OUT_JSON = f'{BASE}/fm_v2ft_results.json'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROIS = [
    ("Cuneus",                        "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",           "Heschl.pth",    "Heschl's Gyrus"),
    ("Middle frontal gyrus anterior", "Midfront.pth",  "Mid. Frontal"),
    ("Precuneus anterior",            "Precuneus.pth", "Precuneus Ant."),
    ("Putamen",                       "Putamen.pth",   "Putamen"),
    ("Thalamus",                       "Thalamus.pth",  "Thalamus"),
    ("global signal clean",           "glb.pth",       "Global Signal"),
]
ROI_DISPLAY = [r[2] for r in ROIS]

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'

def to_float(x): return float(np.asarray(x).flat[0])

def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone with partial fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def build_backbone_finetune(ckpt_path, unfreeze_last_n=2):
    """Load backbone, freeze all but last N transformer blocks + MSS module."""
    m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                     drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=True, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=True)
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = st['model'] if 'model' in st else st
    for k in ['head.weight','head.bias']:
        if k in st and st[k].shape != m.state_dict()[k].shape: del st[k]
    for k in list(st.keys()):
        if 'relative_position_index' in k: st.pop(k)
    m.load_state_dict(st, strict=False)

    # Freeze everything first
    for p in m.parameters():
        p.requires_grad_(False)

    # Unfreeze last N transformer blocks
    n_blocks = len(m.blocks)
    for i in range(n_blocks - unfreeze_last_n, n_blocks):
        for p in m.blocks[i].parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

    # Unfreeze MSS module
    for p in m.mss_module.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f'  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    m.to(DEVICE)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (reuse from FM v2)
# ─────────────────────────────────────────────────────────────────────────────
def load_scan(sub, scan, roi_col):
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')
    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200: raw.resample(200)
    raw.load_data(); raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    df = pd.read_pickle(fm_path)
    fm = df[[roi_col]].to_numpy().T
    b, a = butter(5, 0.15/(0.5/TR), btype='low')
    fm = filtfilt(b, a, fm, axis=1)
    fm, _ = preproc.normalize_data(fm)
    data_epoch, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
    del raw
    n = len(data_epoch["eeg"])
    traincrop = int(0.8*n)
    valcrop   = traincrop + int(0.1*n) + math.ceil(20/TR)
    eeg_all  = torch.stack([torch.tensor(x, dtype=torch.float32) for x in data_epoch["eeg"]])
    tgt_all  = torch.tensor([to_float(x) for x in data_epoch["fmri"]], dtype=torch.float32)
    return eeg_all[:valcrop], tgt_all[:valcrop], eeg_all[valcrop:], tgt_all[valcrop:]


# ─────────────────────────────────────────────────────────────────────────────
# FM head (same as FM v2)
# ─────────────────────────────────────────────────────────────────────────────
class FMHeadV2FT(nn.Module):
    def __init__(self, feat_dim=200, hidden=512, time_dim=32):
        super().__init__()
        freqs = torch.pow(10000, -torch.arange(0, time_dim//2).float() / (time_dim//2))
        self.register_buffer('freqs', freqs)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, hidden//4), nn.SiLU(),
            nn.Linear(hidden//4, 1))

    def time_embed(self, t):
        angles = t * self.freqs[None, :]
        te = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.time_mlp(te)

    def forward(self, feat, x, t):
        te = self.time_embed(t)
        return self.net(torch.cat([feat, x, te], dim=-1))

    @torch.no_grad()
    def sample(self, feat, n_samples=20, steps=100):
        N = len(feat)
        preds = []
        for _ in range(n_samples):
            x = torch.randn(N, 1, device=feat.device)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((N, 1), i/steps, device=feat.device)
                x = x + self(feat, x, t) * dt
            preds.append(x.cpu().squeeze(-1).numpy())
        return np.mean(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Extract features from potentially-unfrozen backbone (needs grad)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_grad(backbone, eeg_tensor, bs=64):
    """Extract features, maintaining computation graph for frozen backbone."""
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b,'B N A T -> B N (A T)'), input_chans=None)
        feat = backbone.head_act(xm + xt)
        out.append(feat.detach().cpu())
    return torch.cat(out)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end training with partial backbone fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def train_fm_v2ft(roi_col, ckpt_fname, display, epochs=300):
    print(f'\n{"="*60}')
    print(f'Training FM v2+FT for ROI: {display}')
    print(f'{"="*60}')

    ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
    backbone = build_backbone_finetune(ckpt_path, unfreeze_last_n=2)

    # Load all data
    eeg_train_list, tgt_train_list = [], []
    eeg_test_list,  tgt_test_list  = [], []
    scan_names = []

    for (sub, scan) in ALL_SCANS:
        try:
            et, tt, ete, tte = load_scan(sub, scan, roi_col)
            if len(ete) < 5: continue
            eeg_train_list.append(et)
            tgt_train_list.append(tt)
            eeg_test_list.append(ete)
            tgt_test_list.append(tte)
            scan_names.append(f'sub{sub:02d}-scan{scan:02d}')
        except Exception as e:
            pass

    print(f'  Scans: {len(scan_names)}  Train EEG batches: {len(eeg_train_list)}')

    # Pre-extract features for frozen training first (Phase 1: extract only)
    print('  Phase 1: Pre-extracting features (frozen backbone)...')
    with torch.no_grad():
        feat_train = torch.cat([extract_features_grad(backbone, e) for e in eeg_train_list])
        feat_test  = torch.cat([extract_features_grad(backbone, e) for e in eeg_test_list])
        tgt_train  = torch.cat(tgt_train_list)
        tgt_test   = torch.cat(tgt_test_list)

    print(f'  Features: train={feat_train.shape}, test={feat_test.shape}')

    # Normalize targets
    mu  = tgt_train.mean().item()
    sig = tgt_train.std().item() + 1e-8
    tgt_train_norm = (tgt_train - mu) / sig

    # FM head
    fm_head = FMHeadV2FT(feat_dim=200).to(DEVICE)

    # Phase 2: Train FM head with fixed features (warm up)
    print('  Phase 2: Training FM head on frozen features (150 epochs)...')
    opt_head = torch.optim.AdamW(fm_head.parameters(), lr=3e-4, weight_decay=1e-4)
    bs = 128
    steps_per_ep = max(1, math.ceil(len(feat_train) / bs))
    sch_head = torch.optim.lr_scheduler.OneCycleLR(opt_head, max_lr=3e-4, total_steps=150*steps_per_ep)

    for ep in range(150):
        fm_head.train(); backbone.eval()
        perm = torch.randperm(len(feat_train))
        for i in range(0, len(feat_train), bs):
            idx = perm[i:i+bs]
            x1 = tgt_train_norm[idx].unsqueeze(-1).to(DEVICE)
            f  = feat_train[idx].to(DEVICE)
            x0 = torch.randn_like(x1)
            t  = torch.rand(len(idx), 1, device=DEVICE)
            xt = (1-t)*x0 + t*x1
            vt = x1 - x0
            loss = F.mse_loss(fm_head(f, xt, t), vt)
            opt_head.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm_head.parameters(), 1.0)
            opt_head.step(); sch_head.step()
        if (ep+1) % 50 == 0:
            print(f'    ep {ep+1}/150  loss={loss.item():.4f}')

    # Phase 3: End-to-end fine-tuning (backbone unfrozen)
    print('  Phase 3: End-to-end fine-tuning (last 2 blocks + FM head, 150 epochs)...')
    del feat_train  # Will re-extract on-the-fly during E2E training
    gc.collect(); torch.cuda.empty_cache()

    # Small batch E2E: process raw EEG → features → FM loss
    # Keep EEG in CPU, load batch to GPU
    eeg_all  = torch.cat(eeg_train_list)  # (N_train, 26, 3200)
    tgt_all  = tgt_train  # already on cpu

    tgt_norm = (tgt_all - mu) / sig

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    opt_e2e = torch.optim.AdamW([
        {'params': fm_head.parameters(), 'lr': 3e-4},
        {'params': backbone_params, 'lr': 5e-6}  # tiny LR for backbone
    ], weight_decay=1e-4)
    sch_e2e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e2e, T_max=150*steps_per_ep)

    ic = get_input_chans()
    bs_e2e = 64  # smaller batch for E2E (backbone needs memory)

    for ep in range(150):
        fm_head.train(); backbone.train()
        perm = torch.randperm(len(eeg_all))
        for i in range(0, len(eeg_all), bs_e2e):
            idx = perm[i:i+bs_e2e]
            eeg_b = eeg_all[idx].to(DEVICE) / 100.
            eeg_b = rearrange(eeg_b, 'B N (A T) -> B N A T', T=200)
            # Extract features (with grad for backbone)
            xt_f = backbone.forward_ts_features(eeg_b, input_chans=ic)
            xm_f = backbone.mss_module(rearrange(eeg_b,'B N A T -> B N (A T)'), input_chans=None)
            feat = backbone.head_act(xm_f + xt_f)  # (B, 200) — grad flows through backbone

            x1 = tgt_norm[idx].unsqueeze(-1).to(DEVICE)
            x0 = torch.randn_like(x1)
            t  = torch.rand(len(idx), 1, device=DEVICE)
            xt = (1-t)*x0 + t*x1
            vt = x1 - x0
            loss = F.mse_loss(fm_head(feat, xt, t), vt)

            opt_e2e.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fm_head.parameters()) + backbone_params, 1.0)
            opt_e2e.step(); sch_e2e.step()

        if (ep+1) % 50 == 0:
            print(f'    ep {ep+1}/150  e2e_loss={loss.item():.4f}')

    # ── Evaluation ────────────────────────────────────────────────────────────
    print('  Evaluating...')
    fm_head.eval(); backbone.eval()
    with torch.no_grad():
        feat_test_new = extract_features_grad(backbone, torch.cat(eeg_test_list))

    pred_norm = fm_head.sample(feat_test_new.to(DEVICE), n_samples=20)
    pred = pred_norm * sig + mu
    true = tgt_test.numpy()

    # Global R
    r_global, _ = pearsonr(pred, true)
    mse_global = float(np.mean((pred - true)**2))
    print(f'  FM v2+FT Global: R={r_global:.3f}  MSE={mse_global:.3f}')

    # Per-scan R
    per_scan_rs = []
    ptr = 0
    for ete in eeg_test_list:
        n = len(ete)
        scan_pred = pred[ptr:ptr+n]
        scan_true = true[ptr:ptr+n]
        if len(scan_pred) >= 5:
            r_s, _ = pearsonr(scan_pred, scan_true)
            per_scan_rs.append(float(r_s))
        ptr += n

    # Save checkpoint
    ckpt_save = {
        'head': fm_head.state_dict(),
        'backbone': backbone.state_dict(),
        'mu': mu, 'sig': sig,
        'roi_col': roi_col, 'display': display,
    }
    ckpt_out = os.path.join(FT_DIR, f'fm_v2ft_{ckpt_fname.replace(".pth","")}.pth')
    torch.save(ckpt_save, ckpt_out)
    print(f'  Saved: {ckpt_out}')

    del backbone, fm_head, eeg_all
    gc.collect(); torch.cuda.empty_cache()

    return {
        'R': float(r_global),
        'MSE': float(mse_global),
        'per_scan_Rs': per_scan_rs,
    }


def main():
    print('\n' + '='*70)
    print('FM-NeuroBOLT v2+FT: FM with Partial Backbone Fine-tuning')
    print('='*70)
    print(f'Device: {DEVICE}')
    print('Configuration: unfreeze last 2 LaBraM blocks + MSS module')
    print('Training: 150 epochs frozen + 150 epochs E2E (backbone lr=5e-6, head lr=3e-4)')

    results = {}
    for roi_col, ckpt_fname, display in ROIS:
        results[display] = train_fm_v2ft(roi_col, ckpt_fname, display, epochs=300)

    # Summary
    avg_r = float(np.nanmean([results[d]['R'] for d in ROI_DISPLAY]))
    results['avg_r'] = avg_r

    print('\n' + '='*80)
    print('FM v2+FT Final Results:')
    print('='*80)
    print(f"{'ROI':<22} {'R':>8} {'MSE':>8}")
    print('-'*40)
    for d in ROI_DISPLAY:
        if d in results:
            print(f"{d:<22} {results[d]['R']:>8.3f} {results[d]['MSE']:>8.3f}")
    print(f"\n=> FM v2+FT Avg.R = {avg_r:.3f}")
    print(f"   Compare: NeuroBOLT=0.406, FM v2=0.424, Ridge=0.446")

    # Full table
    print('\n' + '='*100)
    print('  NeuroBOLT Paper Table 1 + FM v2+FT')
    print('='*100)
    header = f"{'Method':<28}"
    for d in ROI_DISPLAY:
        header += f"  {d[:8]:>10}"
    header += f"  {'Avg.R':>8}"
    print(header)
    print('-'*100)

    nb = [0.453, 0.438, 0.369, 0.441, 0.285, 0.378, 0.482]
    ridge = [0.562, 0.561, 0.371, 0.447, 0.269, 0.401, 0.512]
    fm2 = [0.557, 0.547, 0.344, 0.430, 0.235, 0.364, 0.490]
    fm2ft = [results.get(d, {}).get('R', 0) for d in ROI_DISPLAY]

    for name, vals in [('NeuroBOLT (pretrained)', nb),
                       ('Ridge (per-ROI bb)', ridge),
                       ('FM v2 (frozen bb)', fm2),
                       (f'FM v2+FT (2 blocks)', fm2ft)]:
        row = f"{name:<28}"
        for v in vals:
            row += f"  {v:>10.3f}"
        row += f"  {np.nanmean(vals):>8.3f}"
        print(row)
    print('='*100)

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
