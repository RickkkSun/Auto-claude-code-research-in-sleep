"""
FM-NeuroBOLT v3b — Multi-Source Joint 7D Conditional Flow Matching

Key innovation: MULTI-PERSPECTIVE feature fusion + joint 7D OT-CFM.
Each of the 7 ROI-specific NeuroBOLT backbones is a specialized EEG encoder
optimized for its ROI. We concatenate all 7 representations (7×200=1400D)
and train a single joint 7D FM that models the full joint fMRI distribution.

This is more powerful than:
  - v2: 7 independent 1D FMs (ignores cross-ROI covariance)
  - v3: 1 shared backbone → joint FM (loses ROI-specific representations)
  - v3b: 7 specialized backbones → joint FM (best of both worlds)

Architecture:
  EEG → [Backbone₁, ..., Backbone₇] (frozen, specialized per ROI)
  → 7 × 200-dim features → concat → 1400-dim multi-source features
  → Feature projector: 1400 → 512 (shared compression + cross-ROI mixing)
  → Joint 7D OT-CFM: v_θ(features_512, x_t ∈ R^7, t) → velocity ∈ R^7

Memory: 84MB features (vs 5GB raw EEG in v3) — fully feasible

Novelty: "Multi-Perspective Conditional Flow Matching for EEG-to-fMRI"
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

# ─────────────────────────────────────────────────────────────────────────────
ROIS = [
    ("Cuneus",                        "Cuneus.pth",    "Cuneus"),
    ("Heschl\u2019s gyrus",           "Heschl.pth",    "Heschl's Gyrus"),
    ("Middle frontal gyrus anterior", "Midfront.pth",  "Mid. Frontal"),
    ("Precuneus anterior",            "Precuneus.pth", "Precuneus Ant."),
    ("Putamen",                       "Putamen.pth",   "Putamen"),
    ("Thalamus",                       "Thalamus.pth",  "Thalamus"),
    ("global signal clean",           "glb.pth",       "Global Signal"),
]
ROI_COLS = [r[0] for r in ROIS]
ROI_DISPLAY = [r[2] for r in ROIS]
N_ROIS = 7
FEAT_PER_BB = 200   # per-backbone feature dim
FEAT_TOTAL = N_ROIS * FEAT_PER_BB  # 1400

CH_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPZ','FZ','CZ','PZ',
            'POZ','OZ','FT9','FT10','TP9','TP10']
VECTOR_EXCLUDE_LONG  = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG','CWL1','CWL2','CWL3','CWL4']
VECTOR_EXCLUDE_SHORT = ['EOG1','EOG2','EMG1','EMG2','EMG3','ECG']
TR, TMIN, CROP, EVENT = 2.1, -16, 3200, 'R149'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_SCANS = [
    (1,1),(2,1),(3,2),(4,1),(5,1),(6,1),
    (7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,2),
    (11,1),(12,1),(12,2),(13,1),(13,2),(14,1),(14,2),
    (15,1),(16,1),(17,2),(18,1),(19,2),(20,1),(20,2),(21,1),(22,1)
]

CKPT_DIR  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/code/checkpoints'
DATA_ROOT = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/data'
SAVE_DIR  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/checkpoints_fm_v3b'
NB_JSON   = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/neurobolt_intra_results.json'
V1_JSON   = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_results.json'
V2_JSON   = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v2_results.json'
OUT_JSON  = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm/fm_v3_results.json'
os.makedirs(SAVE_DIR, exist_ok=True)


def to_float(x):
    return float(np.asarray(x).flat[0])

def get_input_chans():
    from utils import get_input_chans as _g
    return _g(CH_NAMES)

def build_backbone(ckpt_path):
    m = create_model('neurobolt_default', EEG_channel=26, num_roi=1,
                     drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
                     drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
                     use_rel_pos_bias=True, use_abs_pos_emb=True,
                     init_values=0.1, qkv_bias=True)
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = st['model'] if 'model' in st else st
    for k in ['head.weight','head.bias']:
        if k in st and st[k].shape != m.state_dict()[k].shape:
            del st[k]
    for k in list(st.keys()):
        if 'relative_position_index' in k:
            st.pop(k)
    m.load_state_dict(st, strict=False)
    m.eval().to(DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m

@torch.no_grad()
def extract_features(backbone, eeg_tensor, bs=64):
    ic = get_input_chans()
    out = []
    for i in range(0, len(eeg_tensor), bs):
        b = eeg_tensor[i:i+bs].to(DEVICE) / 100.
        b = rearrange(b, 'B N (A T) -> B N A T', T=200)
        xt = backbone.forward_ts_features(b, input_chans=ic)
        xm = backbone.mss_module(rearrange(b,'B N A T -> B N (A T)'), input_chans=None)
        out.append(backbone.head_act(xm + xt).cpu())
    return torch.cat(out)  # (N, 200)


def load_scan_all_rois(sub, scan):
    """Load EEG + all 7 ROI fMRI targets for one scan."""
    patient  = f'sub{sub:02d}-scan{scan:02d}'
    eeg_path = os.path.join(DATA_ROOT, 'EEG', f'{patient}_eeg.set')
    fm_path  = os.path.join(DATA_ROOT, 'fMRI_difumo64', f'{patient}_difumo64_roi.pkl')

    raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
    excl = VECTOR_EXCLUDE_LONG if len(raw.ch_names) > 32 else VECTOR_EXCLUDE_SHORT
    raw.drop_channels(excl, on_missing='ignore')
    if raw.info['sfreq'] != 200:
        raw.resample(200)
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)

    df = pd.read_pickle(fm_path)
    b_lp, a_lp = butter(5, 0.15/(0.5/TR), btype='low')

    all_fmri = []
    eeg_all = None
    n = None

    for col in ROI_COLS:
        try:
            fm = df[[col]].to_numpy().T
        except KeyError:
            del raw
            return None
        fm = filtfilt(b_lp, a_lp, fm, axis=1)
        fm, _ = preproc.normalize_data(fm)
        ep, _ = preproc.epoching_seq2one(raw, fm, TMIN, 0, EVENT, ifnorm=0, crop=CROP)
        if eeg_all is None:
            n = len(ep["eeg"])
            eeg_all = torch.stack([torch.tensor(x, dtype=torch.float32) for x in ep["eeg"]])
        all_fmri.append(torch.tensor([to_float(x) for x in ep["fmri"]], dtype=torch.float32))

    del raw
    fmri_all = torch.stack(all_fmri, dim=1)  # (n, 7)

    traincrop = int(0.8 * n)
    valcrop   = traincrop + int(0.1 * n) + math.ceil(20 / TR)

    return (eeg_all[:valcrop], fmri_all[:valcrop],
            eeg_all[valcrop:], fmri_all[valcrop:])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Extract multi-source features from all 7 backbones
# ─────────────────────────────────────────────────────────────────────────────
def extract_multi_source_features():
    """
    For each scan, extract features from all 7 specialized backbones.
    Returns:
      feat_train: (N_train, 1400) — concatenated multi-source features
      tgt_train:  (N_train, 7)   — all 7 ROI targets
      feat_test:  (N_test, 1400)
      tgt_test:   (N_test, 7)
      scan_names: list of scan identifiers (for per-scan evaluation)
      test_slices: scan-wise slices into feat_test/tgt_test
    """
    print('\n[1/3] Loading all scans and extracting multi-source features...')
    print('  7 ROI-specific backbones will be used as specialized EEG encoders')

    # First pass: load EEG + fMRI for all scans
    eeg_train_scans = []  # list of (N_train_i, 26, 3200) EEG tensors per scan
    tgt_train_scans = []  # list of (N_train_i, 7) fMRI tensors
    eeg_test_scans  = []
    tgt_test_scans  = []
    scan_names      = []

    for (sub, scan) in ALL_SCANS:
        pat = f'sub{sub:02d}-scan{scan:02d}'
        print(f'  {pat}...', end=' ', flush=True)
        try:
            result = load_scan_all_rois(sub, scan)
            if result is None:
                print('SKIP (missing ROI column)', flush=True)
                continue
            et, tt, ete, tte = result
            if len(ete) < 5:
                print('SKIP (short test)', flush=True)
                continue
            eeg_train_scans.append(et)
            tgt_train_scans.append(tt)
            eeg_test_scans.append(ete)
            tgt_test_scans.append(tte)
            scan_names.append(pat)
            print(f'tr={len(et)} te={len(ete)}', flush=True)
        except Exception as e:
            print(f'SKIP ({e})', flush=True)

    n_scans = len(scan_names)
    print(f'\n  Loaded {n_scans} scans', flush=True)
    print(f'  Total train: {sum(len(x) for x in eeg_train_scans)}', flush=True)
    print(f'  Total test:  {sum(len(x) for x in eeg_test_scans)}', flush=True)

    # Second pass: extract features from each of 7 backbones
    all_feat_train = []  # list of (N_train, 200) tensors, one per backbone
    all_feat_test  = []  # list of (N_test, 200) tensors

    for j, (roi_col, ckpt_fname, display) in enumerate(ROIS):
        ckpt_path = os.path.join(CKPT_DIR, ckpt_fname)
        print(f'\n  Backbone {j+1}/7: {display} ({ckpt_fname})...', flush=True)
        backbone = build_backbone(ckpt_path)

        feat_tr_parts = []
        feat_te_parts = []

        for i, (et, ete) in enumerate(zip(eeg_train_scans, eeg_test_scans)):
            feat_tr_parts.append(extract_features(backbone, et))
            feat_te_parts.append(extract_features(backbone, ete))
            if (i + 1) % 10 == 0:
                print(f'    {i+1}/{n_scans} scans done', flush=True)

        all_feat_train.append(torch.cat(feat_tr_parts, dim=0))   # (N_train, 200)
        all_feat_test.append(torch.cat(feat_te_parts, dim=0))    # (N_test, 200)

        del backbone; gc.collect(); torch.cuda.empty_cache()

    # Free raw EEG from memory (only features needed from now)
    del eeg_train_scans, eeg_test_scans
    gc.collect()

    # Concatenate all backbone features: (N, 7*200) = (N, 1400)
    feat_train = torch.cat(all_feat_train, dim=1)   # (N_train, 1400)
    feat_test  = torch.cat(all_feat_test, dim=1)    # (N_test, 1400)
    tgt_train  = torch.cat(tgt_train_scans, dim=0)  # (N_train, 7)
    tgt_test   = torch.cat(tgt_test_scans, dim=0)   # (N_test, 7)

    # Compute per-scan slices for evaluation
    test_slices = []
    idx = 0
    for tte in tgt_test_scans:
        n = len(tte)
        test_slices.append(slice(idx, idx+n))
        idx += n

    print(f'\n  Multi-source feature shape: train={feat_train.shape}, test={feat_test.shape}', flush=True)
    print(f'  Feature memory: {feat_train.nbytes / 1e6:.1f} MB train, {feat_test.nbytes / 1e6:.1f} MB test', flush=True)

    return feat_train, tgt_train, feat_test, tgt_test, scan_names, test_slices


# ─────────────────────────────────────────────────────────────────────────────
# Joint 7D FM Head with cross-source attention
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MultiSourceFMHead(nn.Module):
    """
    Joint 7D FM head for multi-source (7-backbone) features.

    Takes:
      - 1400-dim multi-source features (7 backbones × 200D each)
      - x_t ∈ R^7 (current noisy state)
      - t ∈ [0,1] (diffusion time)

    Returns: velocity ∈ R^7

    Architecture:
      Feature projector: 1400 → 512 (compresses + mixes across sources)
      FM head: (512 + 7 + 32) → 512 → 256 → 128 → 7
    """
    def __init__(self, feat_dim=1400, proj_dim=512, hidden_dim=512,
                 n_rois=7, time_dim=32):
        super().__init__()
        self.n_rois = n_rois
        self.time_emb = SinusoidalEmbedding(time_dim)

        # Feature projector: compress 1400D → 512D with cross-source mixing
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
        )

        # Cross-ROI interaction layer for x_t
        self.roi_interact = nn.Linear(n_rois, n_rois)

        # FM velocity network
        in_dim = proj_dim + n_rois + time_dim
        self.velocity_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_rois),
        )

    def forward(self, features, x_t, t):
        """
        features: (B, 1400) multi-source features
        x_t:      (B, 7)   noisy state
        t:        (B, 1)   time
        """
        f_proj = self.feat_proj(features)          # (B, 512)
        t_emb  = self.time_emb(t)                  # (B, 32)
        x_t_i  = self.roi_interact(x_t)            # (B, 7) cross-ROI coupling

        inp = torch.cat([f_proj, x_t_i, t_emb], dim=-1)
        return self.velocity_net(inp)

    @torch.no_grad()
    def sample(self, features, n_samples=20, num_steps=100):
        """Monte-Carlo mean: average N independent 7D ODE trajectories."""
        B = features.shape[0]
        preds = []
        for _ in range(n_samples):
            x = torch.randn(B, self.n_rois, device=features.device)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i / num_steps, device=features.device)
                x = x + self.forward(features, x, t) * dt
            preds.append(x)
        return torch.stack(preds).mean(0)  # (B, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Training: Joint 7D OT-CFM on multi-source features
# ─────────────────────────────────────────────────────────────────────────────
def train_joint_fm(feat_train, tgt_train, epochs=300, lr=3e-4, bs=128, n_samples=20):
    """Train joint 7D OT-CFM on pre-extracted 1400-dim multi-source features."""
    N = len(feat_train)
    print(f'  Training samples: {N}', flush=True)

    # Z-score normalize each ROI independently
    mu  = tgt_train.mean(0)    # (7,)
    sig = tgt_train.std(0) + 1e-8
    tgt_norm = (tgt_train - mu) / sig  # (N, 7)

    head = MultiSourceFMHead(
        feat_dim=FEAT_TOTAL, proj_dim=512, hidden_dim=512, n_rois=N_ROIS
    ).to(DEVICE)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(N / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch
    )

    print(f'  Steps/epoch={steps_per_epoch}, total={epochs*steps_per_epoch}', flush=True)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            c  = feat_train[idx].to(DEVICE)        # (B, 1400)
            x1 = tgt_norm[idx].to(DEVICE)          # (B, 7)

            # OT-CFM: straight-line path x0 → x1
            t   = torch.rand(len(c), 1, device=DEVICE)
            x0  = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            target_vel = x1 - x0

            v_pred = head(c, x_t, t)
            loss   = F.mse_loss(v_pred, target_vel)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sch.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 50 == 0:
            print(f'    ep {ep+1}/{epochs}  loss={epoch_loss/n_batches:.4f}', flush=True)

    return head, mu.numpy(), sig.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(head, feat_test, tgt_test, scan_names, test_slices, mu, sig, n_samples=20):
    """Evaluate joint 7D model on test set."""
    head.eval()
    print('\n[3/3] Evaluation:', flush=True)

    # Full joint inference
    with torch.no_grad():
        pred_norm = head.sample(feat_test.to(DEVICE), n_samples=n_samples, num_steps=100)
    pred_norm = pred_norm.cpu().numpy()  # (N_test, 7)
    pred = pred_norm * sig[None, :] + mu[None, :]  # de-normalize

    true = tgt_test.numpy()

    # Per-scan mean R
    for nm, sl in zip(scan_names, test_slices):
        p_s = pred[sl]; t_s = true[sl]
        if len(p_s) >= 5:
            scan_rs = [pearsonr(p_s[:, j], t_s[:, j])[0] for j in range(N_ROIS)]
            print(f'  {nm}: mean_R={np.mean(scan_rs):.3f}', flush=True)

    # Global R per ROI
    roi_results = {}
    per_scan_rs = {d: [] for d in ROI_DISPLAY}

    for j, disp in enumerate(ROI_DISPLAY):
        all_p = pred[:, j]; all_t = true[:, j]
        r_g, _ = pearsonr(all_p, all_t)
        mse_g  = float(np.mean((all_p - all_t)**2))
        roi_results[disp] = {'R': float(r_g), 'MSE': float(mse_g)}

        # Per-scan Rs
        for sl in test_slices:
            p_s = pred[sl, j]; t_s = true[sl, j]
            if len(p_s) >= 5:
                per_scan_rs[disp].append(float(pearsonr(p_s, t_s)[0]))

        print(f'  {disp}: Global R={r_g:.3f}  MSE={mse_g:.3f}', flush=True)

    avg_r = float(np.nanmean([roi_results[d]['R'] for d in ROI_DISPLAY]))
    print(f'\n  => FM v3b Avg.R = {avg_r:.3f}  (NeuroBOLT: 0.406, FM v2: 0.424)', flush=True)

    # Covariance recovery
    pred_corr = np.corrcoef(pred.T)  # (7, 7)
    true_corr = np.corrcoef(true.T)  # (7, 7)
    cov_mae = float(np.mean(np.abs(pred_corr - true_corr)[~np.eye(N_ROIS, dtype=bool)]))
    print(f'  Cross-ROI covariance MAE: {cov_mae:.4f}', flush=True)

    roi_results['avg_r'] = avg_r
    roi_results['cov_mae'] = cov_mae
    roi_results['per_scan_Rs'] = per_scan_rs
    return roi_results


# ─────────────────────────────────────────────────────────────────────────────
# Table printing (NeuroBOLT paper format)
# ─────────────────────────────────────────────────────────────────────────────
NB_KEY_MAP = {
    'Cuneus': 'Cuneus', "Heschl's Gyrus": 'Heschl', 'Mid. Frontal': 'MidFrontal',
    'Precuneus Ant.': 'PrecuneusAnt', 'Putamen': 'Putamen',
    'Thalamus': 'Thalamus', 'Global Signal': 'GlobalSig'
}

def print_table(v3_results):
    with open(NB_JSON) as f: nb = json.load(f)

    v1_data = v2_data = None
    if os.path.exists(V1_JSON):
        with open(V1_JSON) as f: v1_data = json.load(f)
    if os.path.exists(V2_JSON):
        with open(V2_JSON) as f: v2_data = json.load(f)

    cw = 16
    div = '=' * (34 + cw * N_ROIS + 10)
    print(); print(div)
    print('  NeuroBOLT Paper Table 1 + Flow Matching Comparison (Pearson R / MSE)')
    print(div)
    hdr = f"{'Method':<34}" + ''.join(f'{d:>{cw}}' for d in ROI_DISPLAY) + f"{'Avg.R':>10}"
    print(hdr); print('-' * len(hdr))

    def print_row(name, data_dict, key_map=None):
        rv   = [data_dict.get(key_map[d] if key_map else d, {}).get('R', float('nan'))   for d in ROI_DISPLAY]
        mv   = [data_dict.get(key_map[d] if key_map else d, {}).get('MSE', float('nan')) for d in ROI_DISPLAY]
        avg  = float(np.nanmean(rv))
        line = f'{name:<34}'
        for r, m in zip(rv, mv):
            line += f'{f"{r:.3f}/{m:.3f}":>{cw}}'
        line += f'{avg:>10.3f}'
        print(line)

    print_row('NeuroBOLT (pretrained)', nb, NB_KEY_MAP)
    if v1_data:
        print_row('FM v1 (frozen, 70%)', v1_data)
    if v2_data:
        print_row('FM v2 (per-ROI bb, 90%)', v2_data)
    print_row('FM v3b (multi-src 7D, 90%)', v3_results)

    print(div)
    print()
    print('FM v3b: 7 specialized backbones → 1400D features → joint 7D OT-CFM')
    print('       N=20 MC inference, 300 epochs, Z-score normalized targets')
    print(f"Covariance recovery MAE: {v3_results.get('cov_mae', float('nan')):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f'\n{"="*60}')
    print(f'FM-NeuroBOLT v3b — Multi-Source Joint 7D Flow Matching')
    print(f'{"="*60}')
    print(f'Device: {DEVICE}')
    print(f'Architecture: 7 specialized backbones → 1400D → joint 7D OT-CFM')
    print(f'N_ROIS={N_ROIS}, FEAT_TOTAL={FEAT_TOTAL}')

    # Phase 1: Extract multi-source features
    feat_train, tgt_train, feat_test, tgt_test, scan_names, test_slices = \
        extract_multi_source_features()

    # Phase 2: Train joint 7D FM
    print('\n[2/3] Training joint 7D multi-source FM...', flush=True)
    head, mu, sig = train_joint_fm(feat_train, tgt_train, epochs=300, lr=3e-4, bs=128)

    # Phase 3: Evaluate
    roi_results = evaluate(head, feat_test, tgt_test, scan_names, test_slices,
                           mu, sig, n_samples=20)

    # Save checkpoint
    save_path = os.path.join(SAVE_DIR, 'fm_v3b_joint7d_multisrc.pth')
    torch.save({
        'head': head.state_dict(),
        'mu':  mu.tolist(),
        'sig': sig.tolist(),
        'roi_display': ROI_DISPLAY,
    }, save_path)
    print(f'\nSaved -> {save_path}')

    # Save results
    v3_save = {d: {'R': roi_results[d]['R'], 'MSE': roi_results[d]['MSE'],
                   'per_scan_Rs': roi_results['per_scan_Rs'][d]}
               for d in ROI_DISPLAY}
    v3_save['avg_r']   = roi_results['avg_r']
    v3_save['cov_mae'] = roi_results['cov_mae']

    with open(OUT_JSON, 'w') as f:
        json.dump(v3_save, f, indent=2)
    print(f'Saved -> {OUT_JSON}')

    # Print paper table
    print_table(roi_results)


if __name__ == '__main__':
    main()
