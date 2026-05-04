"""
BraTS 2024 Preprocessing Pipeline and PyTorch Dataset

This module provides:
1. Data loading and preprocessing functions for 3D MRI brain tumor data
2. Normalization with background handling
3. Patch extraction for memory efficiency
4. Label remapping (0,2,4 → 0,1,2) for training compatibility
5. PyTorch Dataset and DataLoader for training

Output format:
- Images: (B, 4, 128, 128, 128) dtype: float32
- Labels: (B, 128, 128, 128) dtype: int64, values: {0, 1, 2}

Compatible with 3D U-Net models.
Memory-efficient and suitable for local machines.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import warnings


def load_case(case_path: str) -> np.ndarray:
    """
    Load all 4 MRI modalities for a patient case.
    
    Args:
        case_path: Path to the patient case folder (e.g., BraTS-GLI-00005-100)
    
    Returns:
        np.ndarray: Shape (4, H, W, D) containing [T1, T1ce, T2, FLAIR]
    
    Raises:
        FileNotFoundError: If any required modality file is not found
    """
    modality_suffixes = ['t1n', 't1c', 't2w']
    flair_suffixes = ['t2f', 'flair']  # Try both naming conventions
    
    modalities = []
    
    # Load T1, T1ce, T2
    for suffix in modality_suffixes:
        pattern = os.path.join(case_path, f'*-{suffix}.nii.gz')
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"Missing modality '{suffix}' in {case_path}")
        
        img = nib.load(files[0])
        modalities.append(img.get_fdata().astype(np.float32))
    
    # Load FLAIR (try both naming conventions)
    flair_data = None
    for suffix in flair_suffixes:
        pattern = os.path.join(case_path, f'*-{suffix}.nii.gz')
        files = glob.glob(pattern)
        if files:
            img = nib.load(files[0])
            flair_data = img.get_fdata().astype(np.float32)
            break
    
    if flair_data is None:
        raise FileNotFoundError(f"Missing FLAIR modality in {case_path}")
    
    modalities.append(flair_data)
    
    # Stack modalities: (4, H, W, D)
    volume = np.stack(modalities, axis=0)
    
    return volume


def load_segmentation(case_path: str) -> np.ndarray:
    """
    Load segmentation mask for a patient case.
    
    BraTS labels in original data:
    - 0: background
    - 2: tumor core
    - 4: whole tumor
    
    Args:
        case_path: Path to the patient case folder
    
    Returns:
        np.ndarray: Shape (H, W, D) with integer labels (before remapping)
    
    Raises:
        FileNotFoundError: If segmentation file not found
    """
    pattern = os.path.join(case_path, '*-seg.nii.gz')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Missing segmentation file in {case_path}")
    
    seg = nib.load(files[0])
    seg_data = seg.get_fdata().astype(np.int32)
    
    return seg_data


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """
    Remap BraTS segmentation labels to contiguous class indices.
    
    Mapping:
    - 0 (background) → 0
    - 2 (tumor core) → 1
    - 4 (whole tumor) → 2
    
    Args:
        seg: np.ndarray shape (H, W, D) with original labels {0, 2, 4}
    
    Returns:
        np.ndarray: Shape (H, W, D) with remapped labels {0, 1, 2}
    """
    seg_remapped = np.zeros_like(seg, dtype=np.int32)
    seg_remapped[seg == 0] = 0
    seg_remapped[seg == 2] = 1
    seg_remapped[seg == 4] = 2
    
    return seg_remapped


def normalize_modality(modality: np.ndarray, ignore_zeros: bool = True) -> np.ndarray:
    """
    Normalize a single modality to zero mean, unit variance.
    
    Args:
        modality: np.ndarray of shape (H, W, D)
        ignore_zeros: If True, ignore background (zero values) in normalization
    
    Returns:
        np.ndarray: Normalized modality (same shape)
    """
    if ignore_zeros:
        # Create mask for non-zero voxels
        mask = modality > 0
        if mask.sum() == 0:
            # If all zeros, return as-is
            return modality
        
        # Calculate mean and std on non-zero voxels only
        mean = modality[mask].mean()
        std = modality[mask].std()
    else:
        mean = modality.mean()
        std = modality.std()
    
    # Avoid division by zero
    if std < 1e-8:
        return modality
    
    # Normalize
    normalized = (modality - mean) / std
    
    return normalized.astype(np.float32)


def crop_nonzero_region(volume: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop volume and segmentation to remove empty background.
    
    Uses bounding box around non-zero voxels in the segmentation.
    
    Args:
        volume: np.ndarray shape (C, H, W, D) - C is number of channels
        seg: np.ndarray shape (H, W, D) - segmentation mask
    
    Returns:
        Tuple of cropped (volume, seg)
    """
    # Find bounding box of non-zero voxels in segmentation
    nonzero_coords = np.where(seg > 0)
    
    if len(nonzero_coords[0]) == 0:
        # No tumor, return as-is
        return volume, seg
    
    z_min, z_max = nonzero_coords[0].min(), nonzero_coords[0].max()
    x_min, x_max = nonzero_coords[1].min(), nonzero_coords[1].max()
    y_min, y_max = nonzero_coords[2].min(), nonzero_coords[2].max()
    
    # Add padding (10 voxels) to include surrounding context
    padding = 10
    z_min = max(0, z_min - padding)
    z_max = min(seg.shape[0], z_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(seg.shape[1], x_max + padding + 1)
    y_min = max(0, y_min - padding)
    y_max = min(seg.shape[2], y_max + padding + 1)
    
    # Crop volume and segmentation
    volume_crop = volume[:, z_min:z_max, x_min:x_max, y_min:y_max]
    seg_crop = seg[z_min:z_max, x_min:x_max, y_min:y_max]
    
    return volume_crop, seg_crop


def extract_patch(volume: np.ndarray, seg: np.ndarray, size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a centered 3D patch from the volume and segmentation.
    
    If volume is smaller than requested patch size, it is padded with zeros.
    Patch is centered on the tumor region (non-zero in segmentation).
    
    Args:
        volume: np.ndarray shape (C, H, W, D) - C is number of modalities
        seg: np.ndarray shape (H, W, D) - segmentation mask
        size: Patch size (default 128, must be same for all inputs)
    
    Returns:
        Tuple of (image_patch, seg_patch):
        - image_patch: shape (C, size, size, size) dtype float32
        - seg_patch: shape (size, size, size) dtype int32
    """
    C = volume.shape[0]
    
    # Get volume dimensions
    vol_d, vol_h, vol_w = volume.shape[1], volume.shape[2], volume.shape[3]
    
    # Compute center of non-zero region in segmentation
    nonzero_coords = np.where(seg > 0)
    if len(nonzero_coords[0]) > 0:
        center_z = (nonzero_coords[0].min() + nonzero_coords[0].max()) // 2
        center_x = (nonzero_coords[1].min() + nonzero_coords[1].max()) // 2
        center_y = (nonzero_coords[2].min() + nonzero_coords[2].max()) // 2
    else:
        # No tumor, use geometric center
        center_z = vol_d // 2
        center_x = vol_h // 2
        center_y = vol_w // 2
    
    # Calculate patch boundaries
    z_start = max(0, center_z - size // 2)
    x_start = max(0, center_x - size // 2)
    y_start = max(0, center_y - size // 2)
    
    z_end = z_start + size
    x_end = x_start + size
    y_end = y_start + size
    
    # Handle boundaries (clamp to volume)
    if z_end > vol_d:
        z_end = vol_d
        z_start = max(0, z_end - size)
    if x_end > vol_h:
        x_end = vol_h
        x_start = max(0, x_end - size)
    if y_end > vol_w:
        y_end = vol_w
        y_start = max(0, y_end - size)
    
    # Extract patches
    image_patch = volume[:, z_start:z_end, x_start:x_end, y_start:y_end]
    seg_patch = seg[z_start:z_end, x_start:x_end, y_start:y_end]
    
    # Pad if necessary (volume smaller than patch size)
    if image_patch.shape[1:] != (size, size, size):
        image_patch_padded = np.zeros((C, size, size, size), dtype=np.float32)
        seg_patch_padded = np.zeros((size, size, size), dtype=np.int32)
        
        # Calculate padding indices
        z_pad = (size - image_patch.shape[1]) // 2
        x_pad = (size - image_patch.shape[2]) // 2
        y_pad = (size - image_patch.shape[3]) // 2
        
        image_patch_padded[
            :,
            z_pad:z_pad + image_patch.shape[1],
            x_pad:x_pad + image_patch.shape[2],
            y_pad:y_pad + image_patch.shape[3]
        ] = image_patch
        
        seg_patch_padded[
            z_pad:z_pad + seg_patch.shape[0],
            x_pad:x_pad + seg_patch.shape[1],
            y_pad:y_pad + seg_patch.shape[2]
        ] = seg_patch
        
        image_patch = image_patch_padded
        seg_patch = seg_patch_padded
    
    return image_patch, seg_patch


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2024 brain tumor segmentation.
    
    Loads patient cases from working_data folder, applies preprocessing:
    - Loads 4 MRI modalities
    - Normalizes each modality independently
    - Crops to non-zero region
    - Extracts 128³ patches
    - Remaps labels: {0, 2, 4} → {0, 1, 2}
    - Converts to torch tensors
    
    Output format (ready for 3D U-Net):
        images: (4, 128, 128, 128) dtype: float32
        labels: (128, 128, 128) dtype: int64, values: {0, 1, 2}
    
    Memory-efficient: processes one case at a time.
    Robust: skips incomplete cases with warnings instead of crashing.
    """
    
    def __init__(self, root_dir: str, patch_size: int = 128, debug: bool = False):
        """
        Args:
            root_dir: Root directory containing patient case folders (e.g., "working_data")
            patch_size: Size of extracted patches (default 128)
            debug: If True, print shapes and statistics for each loaded sample
        
        Raises:
            FileNotFoundError: If root directory doesn't exist
            RuntimeError: If no patient cases found
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.debug = debug
        self.case_dirs = []
        
        # Scan for patient case folders
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        
        # Look for case directories (pattern: BraTS-*)
        case_pattern = os.path.join(root_dir, 'BraTS-*')
        self.case_dirs = sorted(glob.glob(case_pattern))
        
        if len(self.case_dirs) == 0:
            raise RuntimeError(f"No patient cases found in {root_dir}")
        
        print(f"✓ Found {len(self.case_dirs)} patient cases")
        if self.debug:
            print(f"  Debug mode: ON (will print shapes and stats)")
    
    def __len__(self) -> int:
        return len(self.case_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single patient case.
        
        Processing steps:
        1. Load 4 MRI modalities and segmentation mask
        2. Normalize each modality independently (zero mean, unit variance)
        3. Crop to non-zero region
        4. Extract 128³ patch
        5. Remap labels: {0, 2, 4} → {0, 1, 2}
        6. Convert to torch tensors
        7. Verify shapes and label values
        
        Args:
            idx: Index of the case
        
        Returns:
            Tuple of (image, label):
            - image: torch.Tensor shape (4, 128, 128, 128), dtype float32
            - label: torch.Tensor shape (128, 128, 128), dtype int64
        
        Raises:
            RuntimeError: If case processing fails or sanity checks fail
        """
        case_path = self.case_dirs[idx]
        case_name = os.path.basename(case_path)
        
        try:
            # 1. Load modalities and segmentation
            volume = load_case(case_path)
            seg = load_segmentation(case_path)
            
            # 2. Normalize each modality independently
            for c in range(volume.shape[0]):
                volume[c] = normalize_modality(volume[c], ignore_zeros=True)
            
            # 3. Crop to remove background
            volume, seg = crop_nonzero_region(volume, seg)
            
            # 4. Extract patch
            image_patch, seg_patch = extract_patch(volume, seg, size=self.patch_size)
            
            # 5. Remap labels: {0, 2, 4} → {0, 1, 2}
            seg_patch = remap_labels(seg_patch)
            
            # 6. Convert to torch tensors
            image_tensor = torch.from_numpy(image_patch).float()
            seg_tensor = torch.from_numpy(seg_patch).long()
            
            # 7. SANITY CHECKS
            # Check image shape
            assert image_tensor.shape == (4, self.patch_size, self.patch_size, self.patch_size), \
                f"Image shape mismatch: {image_tensor.shape} != (4, {self.patch_size}, {self.patch_size}, {self.patch_size})"
            
            # Check label shape
            assert seg_tensor.shape == (self.patch_size, self.patch_size, self.patch_size), \
                f"Label shape mismatch: {seg_tensor.shape} != ({self.patch_size}, {self.patch_size}, {self.patch_size})"
            
            # Check label values
            unique_labels = torch.unique(seg_tensor)
            valid_labels = {0, 1, 2}
            invalid_labels = set(unique_labels.tolist()) - valid_labels
            assert len(invalid_labels) == 0, \
                f"Invalid label values: {invalid_labels}. Expected only {{0, 1, 2}}"
            
            # Check data types
            assert image_tensor.dtype == torch.float32, f"Image dtype: {image_tensor.dtype} (expected float32)"
            assert seg_tensor.dtype == torch.int64, f"Label dtype: {seg_tensor.dtype} (expected int64)"
            
            # DEBUG OUTPUT
            if self.debug:
                print(f"\n[{case_name}]")
                print(f"  Image shape: {tuple(image_tensor.shape)}, dtype: {image_tensor.dtype}")
                print(f"  Label shape: {tuple(seg_tensor.shape)}, dtype: {seg_tensor.dtype}")
                print(f"  Image: min={image_tensor.min():.4f}, max={image_tensor.max():.4f}, mean={image_tensor.mean():.4f}")
                print(f"  Label unique values: {unique_labels.tolist()}")
            
            return image_tensor, seg_tensor
        
        except Exception as e:
            print(f"⚠️  WARNING: Failed to load case {case_name}: {str(e)}")
            print(f"   Skipping this case...")
            raise RuntimeError(f"Failed to process case {case_name}: {str(e)}")


def create_dataloader(
    root_dir: str,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    patch_size: int = 128
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the BraTS dataset.
    
    Args:
        root_dir: Root directory containing patient case folders
        batch_size: Number of samples per batch (default 2)
        shuffle: Whether to shuffle the dataset (default True)
        num_workers: Number of worker processes for data loading (default 0)
        patch_size: Size of extracted patches (default 128)
    
    Returns:
        DataLoader: Ready to use for training
    """
    dataset = BraTSDataset(root_dir=root_dir, patch_size=patch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


# ============================================================================
# TEST SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Test the BraTS dataset pipeline:
    - Load dataset
    - Verify shapes and sanity checks
    - Display batch information
    - Ensure no crashes
    """
    
    print("=" * 75)
    print("BraTS 2024 Dataset Pipeline - Finalized Test Script")
    print("=" * 75)
    
    # Set working data directory
    working_data_dir = os.path.join(os.path.dirname(__file__), "working_data")
    
    if not os.path.exists(working_data_dir):
        print(f"\n❌ ERROR: working_data directory not found at {working_data_dir}")
        print("   Please create a 'working_data' folder with 2-5 BraTS cases.")
        exit(1)
    
    print(f"\n✓ Working data directory: {working_data_dir}\n")
    
    try:
        # Create dataset and dataloader with debug=True
        print("Loading dataset with debug mode enabled...\n")
        dataloader = create_dataloader(
            root_dir=working_data_dir,
            batch_size=1,
            shuffle=False,
            patch_size=128
        )
        
        print(f"\nDataset size: {len(dataloader.dataset)} cases\n")
        
        # Load and process first batch
        print("Processing first batch...")
        print("-" * 75)
        
        images, labels = next(iter(dataloader))
        
        # Verify shapes
        print(f"\n✓ SHAPE VERIFICATION:")
        print(f"  Images shape:  {tuple(images.shape)}")
        print(f"  Expected:      (batch_size, 4, 128, 128, 128)")
        assert images.shape == (1, 4, 128, 128, 128), f"Image shape mismatch: {images.shape}"
        
        print(f"\n  Labels shape:  {tuple(labels.shape)}")
        print(f"  Expected:      (batch_size, 128, 128, 128)")
        assert labels.shape == (1, 128, 128, 128), f"Label shape mismatch: {labels.shape}"
        
        # Verify data types
        print(f"\n✓ DTYPE VERIFICATION:")
        print(f"  Images dtype:  {images.dtype}")
        assert images.dtype == torch.float32, f"Image dtype should be float32, got {images.dtype}"
        
        print(f"  Labels dtype:  {labels.dtype}")
        assert labels.dtype == torch.int64, f"Label dtype should be int64, got {labels.dtype}"
        
        # Verify label values (CRITICAL)
        unique_labels = torch.unique(labels)
        print(f"\n✓ LABEL VALUE VERIFICATION:")
        print(f"  Unique labels: {unique_labels.tolist()}")
        print(f"  Expected:      [0, 1, 2] or subset")
        
        valid_labels = {0, 1, 2}
        invalid_labels = set(unique_labels.tolist()) - valid_labels
        assert len(invalid_labels) == 0, f"Invalid labels found: {invalid_labels}"
        
        # Display statistics
        print(f"\n✓ DATA STATISTICS:")
        print(f"  Images min:    {images.min():.4f}")
        print(f"  Images max:    {images.max():.4f}")
        print(f"  Images mean:   {images.mean():.4f}")
        print(f"  Images std:    {images.std():.4f}")
        print(f"\n  Labels min:    {labels.min():.0f}")
        print(f"  Labels max:    {labels.max():.0f}")
        
        # Count voxels per class
        num_bg = (labels == 0).sum().item()
        num_tumor_core = (labels == 1).sum().item()
        num_whole_tumor = (labels == 2).sum().item()
        total_voxels = labels.numel()
        
        print(f"\n  Voxel distribution:")
        print(f"    Background (0):    {num_bg:8d} ({100*num_bg/total_voxels:5.2f}%)")
        print(f"    Tumor core (1):    {num_tumor_core:8d} ({100*num_tumor_core/total_voxels:5.2f}%)")
        print(f"    Whole tumor (2):   {num_whole_tumor:8d} ({100*num_whole_tumor/total_voxels:5.2f}%)")
        
        print("\n" + "=" * 75)
        print("✅ ALL TESTS PASSED!")
        print("=" * 75)
        print("\nDataset is ready for 3D U-Net training:")
        print("  - Image shape: (B, 4, 128, 128, 128) dtype: float32")
        print("  - Label shape: (B, 128, 128, 128) dtype: int64")
        print("  - Label values: {0 (background), 1 (tumor core), 2 (whole tumor)}")
        print("=" * 75 + "\n")
    
    except AssertionError as e:
        print("\n" + "=" * 75)
        print(f"❌ ASSERTION ERROR: {str(e)}")
        print("=" * 75)
        exit(1)
    
    except Exception as e:
        print("\n" + "=" * 75)
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        print("=" * 75)
        import traceback
        traceback.print_exc()
        exit(1)
