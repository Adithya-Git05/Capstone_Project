"""
segmentation.py
===============
Teammate 3 — Segmentation Architect (U-Net)
BraTS Brain Tumor Segmentation

Architecture : Attention U-Net
Loss         : Combined Dice Loss + Cross-Entropy
Dataset      : BraTS 2021 (4-channel MRI: T1, T1ce, T2, FLAIR)
Target       : 3-class segmentation
               0 = background
               1 = necrotic/non-enhancing tumor core (NCR/NET)
               2 = peritumoral edema (ED)
               3 = enhancing tumor (ET)
               → mapped to BraTS regions:
                 TC (Tumor Core)    = labels {1, 3}
                 WT (Whole Tumor)   = labels {1, 2, 3}
                 ET (Enhancing)     = label  {3}

Usage
-----
# Quick smoke-test on 5 patients (Step 3):
python segmentation.py --mode test --data_dir /path/to/brats --n_patients 5

# Full training:
python segmentation.py --mode train --data_dir /path/to/brats --epochs 100
"""

import os
import sys
import argparse
import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy import ndimage
import requests
import zipfile
import tarfile
import shutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Downloader
# ---------------------------------------------------------------------------

def download_and_extract(url: str, extract_to: str):
    """Downloads a file (zip/tar) from a URL and extracts it."""
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    local_filename = extract_path / "dataset_download"
    
    log.info(f"Downloading dataset from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    log.info("Extracting dataset...")
    if zipfile.is_zipfile(local_filename):
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif local_filename.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(local_filename, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        log.warning("Unknown file format. Attempting to move file instead.")
        # If it's a raw file, we might just leave it there
        
    local_filename.unlink() # Remove the archive after extraction
    log.info(f"Dataset ready at {extract_to}")


# ===========================================================================
# 1.  DATASET
# ===========================================================================

class BraTSDataset(Dataset):
    """
    Loads BraTS 2021 cases.

    Expected directory layout (NIfTI):
        <data_dir>/
            BraTS2021_XXXXX/
                BraTS2021_XXXXX_t1.nii.gz
                BraTS2021_XXXXX_t1ce.nii.gz
                BraTS2021_XXXXX_t2.nii.gz
                BraTS2021_XXXXX_flair.nii.gz
                BraTS2021_XXXXX_seg.nii.gz

    Each volume is cropped to a fixed ROI patch_size for memory efficiency.
    """

    # Modality mapping for different BraTS years
    MODALITY_MAP = {
        2021: {"t1": "t1",  "t1ce": "t1ce", "t2": "t2",  "flair": "flair"},
        2024: {"t1": "t1n", "t1ce": "t1c",  "t2": "t2w", "flair": "t2f"} # Common 2024 suffixes
    }

    def __init__(
        self,
        data_dir: str,
        patient_ids: list[str],
        patch_size: tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        year: int = 2024,
    ):
        self.data_dir   = Path(data_dir)
        self.patient_ids = patient_ids
        self.patch_size  = patch_size
        self.augment     = augment
        self.year        = year
        self.suffixes    = self.MODALITY_MAP.get(year, self.MODALITY_MAP[2024])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid  = self.patient_ids[idx]
        pdir = self.data_dir / pid

        # ── load 4 modalities ───────────────────────────────────────────────
        channels = []
        for mod_key in ["t1", "t1ce", "t2", "flair"]:
            suffix = self.suffixes[mod_key]
            # Try both underscore and hyphen (2024 often uses hyphen)
            path = pdir / f"{pid}_{suffix}.nii.gz"
            if not path.exists():
                path = pdir / f"{pid}-{suffix}.nii.gz"
            
            if not path.exists():
                raise FileNotFoundError(f"Could not find modality {mod_key} for {pid} at {path}")
                
            vol  = nib.load(str(path)).get_fdata(dtype=np.float32)
            vol  = self._normalize(vol)
            channels.append(vol)

        image = np.stack(channels, axis=0)          # (4, H, W, D)

        # ── load segmentation mask ──────────────────────────────────────────
        seg_path = pdir / f"{pid}_seg.nii.gz"
        if not seg_path.exists():
            seg_path = pdir / f"{pid}-seg.nii.gz"
            
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32).astype(np.int64)
        
        # Label remapping (BraTS 2024 often uses {0, 2, 4} or {0, 1, 2, 3})
        # If we see 4, we likely have {1, 2, 4} (2021) or {0, 2, 4} (Capstone style)
        if 4 in np.unique(seg):
            if self.year == 2021:
                seg[seg == 4] = 3 # Map ET to 3
            else:
                # Capstone Project style (2->2, 4->3) to match BraTS metrics
                new_seg = np.zeros_like(seg)
                new_seg[seg == 1] = 1 # NCR
                new_seg[seg == 2] = 2 # ED
                new_seg[seg == 4] = 3 # ET
                seg = new_seg
        elif 3 in np.unique(seg):
            # Standard 2024: 0, 1, 2, 3 -> already correct
            pass

        # ── crop to patch ────────────────────────────────────────────────────
        image, seg = self._random_crop(image, seg)  # (4, ps, ps, ps), (ps, ps, ps)

        # ── optional augmentation ────────────────────────────────────────────
        if self.augment:
            image, seg = self._augment(image, seg)

        return (
            torch.from_numpy(image),
            torch.from_numpy(seg),
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vol: np.ndarray) -> np.ndarray:
        """Z-score normalisation inside brain mask (non-zero voxels)."""
        mask = vol > 0
        if mask.sum() == 0:
            return vol
        mu  = vol[mask].mean()
        sig = vol[mask].std() + 1e-8
        out = np.zeros_like(vol)
        out[mask] = (vol[mask] - mu) / sig
        return out

    def _random_crop(self, image, seg):
        ph, pw, pd = self.patch_size
        _, H, W, D = image.shape

        # Ensure patch fits
        h0 = random.randint(0, max(H - ph, 0))
        w0 = random.randint(0, max(W - pw, 0))
        d0 = random.randint(0, max(D - pd, 0))

        image = image[:, h0:h0+ph, w0:w0+pw, d0:d0+pd]
        seg   = seg[h0:h0+ph, w0:w0+pw, d0:d0+pd]

        # Pad if volume smaller than patch (rare edge case)
        pads = [(0, 0)] + [
            (0, max(0, p - s))
            for p, s in zip(self.patch_size, image.shape[1:])
        ]
        image = np.pad(image, pads)
        seg   = np.pad(seg, [(0, max(0, p - s)) for p, s in zip(self.patch_size, seg.shape)])

        return image, seg

    @staticmethod
    def _augment(image, seg):
        """Light augmentation: random flip along each axis."""
        for axis in range(1, 4):
            if random.random() > 0.5:
                image = np.flip(image, axis=axis).copy()
                seg   = np.flip(seg,   axis=axis - 1).copy()
        return image, seg


def get_patient_ids(data_dir: str, n_patients: int | None = None) -> list[str]:
    root = Path(data_dir)
    ids  = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if n_patients:
        ids = ids[:n_patients]
    return ids


# ===========================================================================
# 2.  MODEL — Attention U-Net (3-D)
# ===========================================================================

class ConvBlock(nn.Module):
    """Two consecutive Conv3d → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive soft-attention gate (Oktay et al., 2018).

    Suppresses irrelevant background activations at skip connections,
    allowing the decoder to focus on salient tumour features.

    g  : gating signal from decoder (coarser scale)
    x  : encoder feature map (skip connection)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        # Align spatial dims (g is upsampled before gate in decoder)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)          # attention map ∈ (0,1)
        return x * psi               # weighted skip features


class AttentionUNet(nn.Module):
    """
    3-D Attention U-Net for BraTS segmentation.

    Input  : (B, 4, 128, 128, 128)  — 4 MRI modalities
    Output : (B, num_classes, 128, 128, 128)  — per-voxel class logits
    """

    def __init__(
        self,
        in_channels:  int = 4,
        num_classes:  int = 4,   # 0=BG, 1=NCR, 2=ED, 3=ET
        base_filters: int = 32,
        dropout:      float = 0.2,
    ):
        super().__init__()
        f = base_filters        # channel multiplier

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, f,    dropout=0.0)
        self.enc2 = ConvBlock(f,           f*2,  dropout=0.0)
        self.enc3 = ConvBlock(f*2,         f*4,  dropout=dropout)
        self.enc4 = ConvBlock(f*4,         f*8,  dropout=dropout)

        self.pool = nn.MaxPool3d(2)

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(f*8, f*16, dropout=dropout)

        # ── Decoder ────────────────────────────────────────────────────────
        self.up4   = nn.ConvTranspose3d(f*16, f*8,  2, stride=2)
        self.att4  = AttentionGate(F_g=f*8,  F_l=f*8,  F_int=f*4)
        self.dec4  = ConvBlock(f*16, f*8,  dropout=dropout)

        self.up3   = nn.ConvTranspose3d(f*8,  f*4,  2, stride=2)
        self.att3  = AttentionGate(F_g=f*4,  F_l=f*4,  F_int=f*2)
        self.dec3  = ConvBlock(f*8,  f*4,  dropout=dropout)

        self.up2   = nn.ConvTranspose3d(f*4,  f*2,  2, stride=2)
        self.att2  = AttentionGate(F_g=f*2,  F_l=f*2,  F_int=f)
        self.dec2  = ConvBlock(f*4,  f*2,  dropout=0.0)

        self.up1   = nn.ConvTranspose3d(f*2,  f,    2, stride=2)
        self.att1  = AttentionGate(F_g=f,    F_l=f,    F_int=f//2)
        self.dec1  = ConvBlock(f*2,  f,    dropout=0.0)

        # ── Output ──────────────────────────────────────────────────────────
        self.out_conv = nn.Conv3d(f, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder with attention gates
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


# ===========================================================================
# 3.  LOSS FUNCTIONS
# ===========================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice loss over all classes, averaged.

    Critical for BraTS because the tumour (foreground) occupies only
    ~1–3 % of brain voxels — cross-entropy alone would learn to predict
    "all background" and still achieve >97 % pixel accuracy.
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W, D) — raw logits
        targets : (B, H, W, D)    — integer labels
        """
        probs   = F.softmax(logits, dim=1)                       # (B, C, H, W, D)
        targets_oh = F.one_hot(targets, self.num_classes)        # (B, H, W, D, C)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()  # (B, C, H, W, D)

        start_cls = 1 if self.ignore_bg else 0
        dice_sum  = 0.0
        n_cls     = 0

        for c in range(start_cls, self.num_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            intersection = (p * t).sum()
            union        = p.sum() + t.sum()
            dice_sum    += (2.0 * intersection + self.smooth) / (union + self.smooth)
            n_cls       += 1

        return 1.0 - dice_sum / max(n_cls, 1)


class CombinedLoss(nn.Module):
    """
    L = λ_dice * DiceLoss  +  λ_ce * CrossEntropyLoss

    The two losses are complementary:
      • Dice   → handles class imbalance, optimises overlap directly
      • CE     → stable gradients pixel-by-pixel, prevents degenerate solutions
    """

    def __init__(
        self,
        num_classes: int   = 4,
        lambda_dice: float = 0.5,
        lambda_ce:   float = 0.5,
        ignore_bg:   bool  = True,
    ):
        super().__init__()
        self.dice       = DiceLoss(num_classes, ignore_bg=ignore_bg)
        self.ce         = nn.CrossEntropyLoss()
        self.lambda_dice = lambda_dice
        self.lambda_ce   = lambda_ce

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_ce   = self.ce(logits, targets)
        return self.lambda_dice * loss_dice + self.lambda_ce * loss_ce, loss_dice, loss_ce


# ===========================================================================
# 4.  METRICS
# ===========================================================================

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-5,
) -> dict[str, float]:
    """
    Computes Dice, Precision, Sensitivity, and Accuracy for BraTS regions.
    WT = Whole Tumour   (labels 1+2+3)
    TC = Tumour Core    (labels 1+3)
    ET = Enhancing      (label 3)
    """
    metrics = {}
    p_np = preds.cpu().numpy()
    t_np = targets.cpu().numpy()

    # Per-region metrics
    regions = {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3],
    }

    for name, labels in regions.items():
        p = np.isin(p_np, labels).astype(np.float32)
        t = np.isin(t_np, labels).astype(np.float32)

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tn = ((1 - p) * (1 - t)).sum()

        metrics[f"{name}_Dice"]      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        metrics[f"{name}_Precision"] = (tp + smooth) / (tp + fp + smooth)
        metrics[f"{name}_Sensitivity"] = (tp + smooth) / (tp + fn + smooth)
        metrics[f"{name}_Accuracy"]  = (tp + tn) / (tp + tn + fp + fn + smooth)

    return metrics


# ===========================================================================
# 5.  TRAINING LOOP
# ===========================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = total_dice = total_ce = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, ldice, lce = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, ldice, lce = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_dice += ldice.item()
        total_ce   += lce.item()

        if batch_idx % 5 == 0:
            log.info(
                f"  Batch {batch_idx+1}/{len(loader)} | "
                f"loss={loss.item():.4f}  dice={ldice.item():.4f}  ce={lce.item():.4f}"
            )

    n = len(loader)
    return total_loss / n, total_dice / n, total_ce / n


@torch.no_grad()
def validate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    all_scores: list[dict] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, _, _ = criterion(logits, labels)
        else:
            logits = model(images)
            loss, _, _ = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        all_scores.append(compute_metrics(preds, labels))

    # Average across batches
    avg_scores = {k: np.mean([s[k] for s in all_scores]) for k in all_scores[0]}
    return total_loss / len(loader), avg_scores


def train(args):
    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Using CPU — consider reducing patch_size for memory")

    use_amp = device.type == "cuda"

    # ── Data ────────────────────────────────────────────────────────────────
    all_ids = get_patient_ids(args.data_dir, args.n_patients)
    random.seed(args.seed)
    random.shuffle(all_ids)

    split     = max(1, int(0.8 * len(all_ids)))
    train_ids = all_ids[:split]
    val_ids   = all_ids[split:] or all_ids[-1:]   # at least 1 val case

    log.info(f"Train: {len(train_ids)} patients | Val: {len(val_ids)} patients")

    patch = tuple(args.patch_size)

    train_ds = BraTSDataset(args.data_dir, train_ids, patch_size=patch, augment=True, year=args.dataset_year)
    val_ds   = BraTSDataset(args.data_dir, val_ids,   patch_size=patch, augment=False, year=args.dataset_year)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = AttentionUNet(
        in_channels=4,
        num_classes=4,
        base_filters=args.base_filters,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Attention U-Net | trainable params: {n_params:,}")

    # ── Loss, Optimiser, Scheduler ──────────────────────────────────────────
    criterion = CombinedLoss(num_classes=4, lambda_dice=0.5, lambda_ce=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Checkpoint dir ───────────────────────────────────────────────────────
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_wt = 0.0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch}/{args.epochs}  |  LR = {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_dice, train_ce = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_scores = validate(model, val_loader, criterion, device, scaler)

        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            f"[Train] loss={train_loss:.4f}  dice={train_dice:.4f}  ce={train_ce:.4f}"
        )
        log.info(
            f"[Val]   loss={val_loss:.4f} | WT Metrics: "
            f"Dice={val_scores['WT_Dice']:.4f}  "
            f"Prec={val_scores['WT_Precision']:.4f}  "
            f"Sens={val_scores['WT_Sensitivity']:.4f}  "
            f"Acc={val_scores['WT_Accuracy']:.4f}"
        )
        log.info(
            f"        TC Dice={val_scores['TC_Dice']:.4f}  "
            f"ET Dice={val_scores['ET_Dice']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Save best checkpoint (by Whole Tumour Dice)
        if val_scores["WT_Dice"] > best_wt:
            best_wt = val_scores["WT_Dice"]
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_scores":  val_scores,
                    "args":        vars(args),
                },
                ckpt_path,
            )
            log.info(f"  ✓ Best checkpoint saved  (WT Dice = {best_wt:.4f})")

    log.info(f"\nTraining complete.  Best WT Dice = {best_wt:.4f}")
    return model


# ===========================================================================
# 6.  SMOKE TEST (Step 3 — 5 patients, forward + backward pass)
# ===========================================================================

def smoke_test(args):
    """
    Verifies that:
      1. Data loads without error for n_patients cases.
      2. Forward pass completes (output shape is correct).
      3. Backward pass completes without memory crash.
      4. Loss values are finite.
    Prints peak GPU/CPU memory usage.
    """
    log.info("=" * 60)
    log.info("SMOKE TEST — 5-patient memory check")
    log.info("=" * 60)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log.info(f"Device: {device}")

    # ── tiny dataset ─────────────────────────────────────────────────────────
    all_ids  = get_patient_ids(args.data_dir, args.n_patients)
    patch    = tuple(args.patch_size)
    dataset  = BraTSDataset(args.data_dir, all_ids, patch_size=patch, year=args.dataset_year)
    loader   = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model     = AttentionUNet(in_channels=4, num_classes=4,
                              base_filters=args.base_filters).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    log.info(f"Loaded {len(dataset)} patients | Patch size: {patch}")

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        log.info(f"\nPatient {i+1}: image={tuple(images.shape)}  label={tuple(labels.shape)}")

        # Forward
        optimizer.zero_grad()
        t0     = time.time()
        logits = model(images)
        t_fwd  = time.time() - t0

        assert logits.shape == (images.shape[0], 4, *patch), \
            f"Output shape mismatch: {logits.shape}"

        # Loss + Backward
        loss, ldice, lce = criterion(logits, labels)
        assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"

        t0 = time.time()
        loss.backward()
        optimizer.step()
        t_bwd = time.time() - t0

        log.info(
            f"  Forward {t_fwd:.2f}s  |  Backward {t_bwd:.2f}s  |  "
            f"loss={loss.item():.4f}  dice={ldice.item():.4f}  ce={lce.item():.4f}"
        )

        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            log.info(f"  Peak GPU memory: {mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)

    log.info("\n✅  Smoke test PASSED — forward & backward passes complete for all patients.")


# ===========================================================================
# 7.  INFERENCE / PREDICTION
# ===========================================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    image_paths: dict[str, str],
    patch_size: tuple = (128, 128, 128),
    device: str = "cpu",
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Sliding-window inference on a single BraTS case.

    image_paths : {"t1": path, "t1ce": path, "t2": path, "flair": path}
    Returns     : predicted segmentation volume (H, W, D)  np.int64
    """
    device = torch.device(device)
    model.eval().to(device)

    # Load + normalise channels
    channels = []
    for mod in ["t1", "t1ce", "t2", "flair"]:
        vol = nib.load(image_paths[mod]).get_fdata(dtype=np.float32)
        vol = BraTSDataset._normalize(vol)
        channels.append(vol)

    image = np.stack(channels, axis=0)     # (4, H, W, D)
    _, H, W, D = image.shape
    ph, pw, pd = patch_size

    pred_vol  = np.zeros((4, H, W, D), dtype=np.float32)
    count_vol = np.zeros((H, W, D),    dtype=np.float32)

    step = [max(1, int(p * (1 - overlap))) for p in patch_size]

    h_starts = list(range(0, max(H - ph + 1, 1), step[0]))
    w_starts = list(range(0, max(W - pw + 1, 1), step[1]))
    d_starts = list(range(0, max(D - pd + 1, 1), step[2]))

    for h0 in h_starts:
        for w0 in w_starts:
            for d0 in d_starts:
                h1, w1, d1 = h0 + ph, w0 + pw, d0 + pd
                patch = image[:, h0:h1, w0:w1, d0:d1]
                if patch.shape[1:] != patch_size:
                    continue
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(t)
                probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_vol[:, h0:h1, w0:w1, d0:d1] += probs
                count_vol[h0:h1, w0:w1, d0:d1]   += 1.0

    count_vol = np.maximum(count_vol, 1.0)
    pred_vol /= count_vol[np.newaxis]
    seg = pred_vol.argmax(axis=0).astype(np.int64)
    # Restore BraTS label 3→4
    seg[seg == 3] = 4

    return seg


# ===========================================================================
# 8.  CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="BraTS Attention U-Net Segmentation")

    p.add_argument("--mode",        choices=["train", "test"], default="test",
                   help="'test' = smoke-test on n_patients; 'train' = full run")
    p.add_argument("--data_dir",    default="./data",
                   help="Root directory containing BraTS patient folders")
    p.add_argument("--data_url",    default=None,
                   help="GitHub or URL to download the dataset from (ZIP/TAR)")
    p.add_argument("--output_dir",  default="./checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--n_patients",  type=int,   default=5,
                   help="Limit dataset to first N patients (smoke test)")
    p.add_argument("--patch_size",  type=int,   nargs=3, default=[128, 128, 128],
                   help="3-D training patch size (H W D)")
    p.add_argument("--base_filters",type=int,   default=32,
                   help="Base feature-map width (double at each encoder stage)")
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Batch size (keep 1 for 128³ patches on 16 GB GPU)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--dataset_year", type=int,  default=2024, choices=[2021, 2024])
    p.add_argument("--seed",        type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Handle Remote Data ──────────────────────────────────────────────────
    if args.data_url:
        download_and_extract(args.data_url, args.data_dir)

    if args.mode == "test":
        smoke_test(args)
    else:
        train(args)
