# BraTS 2024 Brain Tumor Segmentation - Capstone Project

A complete preprocessing pipeline and PyTorch Dataset for brain tumor segmentation using the BraTS 2024 dataset, with 3D U-Net compatibility.

## Project Overview

This project implements:
- **Data Preprocessing**: Loads 4 MRI modalities (T1, T1ce, T2, FLAIR)
- **Normalization**: Zero mean, unit variance per modality
- **Spatial Processing**: Automatic cropping and patch extraction
- **Label Management**: Proper remapping of BraTS labels (0,2,4 → 0,1,2)
- **PyTorch Integration**: Ready-to-use Dataset and DataLoader

## Output Format

```
Images: (B, 4, 128, 128, 128) dtype: float32
Labels: (B, 128, 128, 128) dtype: int64 (values: {0, 1, 2})
```

## Installation

### Prerequisites
- Python 3.13+
- Windows PowerShell 5.1+

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Adithya-Git05/Capstone_Project.git
cd Capstone_Project
```

2. **Create virtual environment**:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Pipeline

```python
from dataset import create_dataloader

# Create DataLoader
dataloader = create_dataloader(
    root_dir="working_data",
    batch_size=2,
    shuffle=True,
    patch_size=128
)

# Iterate through batches
for images, labels in dataloader:
    # images: (2, 4, 128, 128, 128) dtype: float32
    # labels: (2, 128, 128, 128) dtype: int64
    
    # Use with 3D U-Net model
    output = model(images)
    loss = criterion(output, labels)
```

### Debug Mode

```python
dataset = BraTSDataset(root_dir="working_data", debug=True)
# Prints shapes and statistics for each loaded case
```

## Dataset Setup

### BraTS 2024 Dataset

Download from: [BraTS 2024 - Hugging Face](https://huggingface.co/datasets/Spirit-26/BraTS-2024-Complete)

```bash
# Download dataset (requires ~12GB)
hf download Spirit-26/BraTS-2024-Complete --repo-type dataset --local-dir BraTS-2024-Complete
```

### Working Data

Create `working_data/` with 2-5 sample cases:
```
working_data/
  BraTS-GLI-00005-100/
    BraTS-GLI-00005-100-t1n.nii.gz
    BraTS-GLI-00005-100-t1c.nii.gz
    BraTS-GLI-00005-100-t2w.nii.gz
    BraTS-GLI-00005-100-t2f.nii.gz
    BraTS-GLI-00005-100-seg.nii.gz
  ...
```

## File Structure

```
Capstone_Project/
├── dataset.py          # Main preprocessing pipeline
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore rules
├── BraTS-2024-Complete/  (Not committed - use .gitignore)
└── working_data/         (Not committed - use .gitignore)
```

## Pipeline Steps

### 1. Load
Loads 4 MRI modalities and segmentation mask from .nii.gz files.

### 2. Normalize
Applies zero mean, unit variance normalization per modality (ignoring background).

### 3. Crop
Removes empty background using bounding box around tumor region.

### 4. Extract Patch
Extracts fixed 128³ patch centered on tumor, with zero-padding if needed.

### 5. Remap Labels
Converts BraTS labels {0, 2, 4} → {0, 1, 2}:
- 0: Background
- 1: Tumor core
- 2: Whole tumor

### 6. Convert Tensors
Returns PyTorch tensors ready for model training.

## Testing

Run the test script:
```bash
python dataset.py
```

Expected output:
```
✓ Images shape:  (1, 4, 128, 128, 128)
✓ Labels shape:  (1, 128, 128, 128)
✓ Labels unique: [0, 1, 2]
✓ Voxel distribution: Background (98.70%), Tumor core (1.09%), Whole tumor (0.20%)
```

## Functions Reference

### `load_case(case_path)` 
Loads 4 MRI modalities → (4, H, W, D)

### `load_segmentation(case_path)`
Loads segmentation mask → (H, W, D)

### `normalize_modality(modality)`
Zero mean, unit variance normalization

### `crop_nonzero_region(volume, seg)`
Removes empty background

### `extract_patch(volume, seg, size=128)`
Extracts fixed 128³ patch

### `remap_labels(seg)`
Maps {0, 2, 4} → {0, 1, 2}

### `BraTSDataset(root_dir, patch_size=128, debug=False)`
PyTorch Dataset class

### `create_dataloader(root_dir, batch_size=2, shuffle=True, num_workers=0, patch_size=128)`
Creates PyTorch DataLoader

## Next Steps

1. **Build 3D U-Net Model** - Use DataLoader output directly
2. **Implement Training Loop** - Add loss functions and optimization
3. **Add Validation** - Separate val/test splits
4. **Evaluate Metrics** - Dice score, Hausdorff distance

## Notes

- **Memory Efficient**: Processes data on-the-fly, no caching
- **Production Ready**: Includes error handling and sanity checks
- **Modular Design**: Functions can be used independently
- **Documented**: Comprehensive docstrings and comments

## Contributing

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make changes and test locally
3. Commit with clear messages: `git commit -m "Add feature: description"`
4. Push to remote: `git push origin feature/your-feature`
5. Create Pull Request on GitHub

## Team Members

- Teammate 1: Data Preprocessing Pipeline (dataset.py)
- Teammate 2: 3D U-Net Model Implementation (pending)
- Teammate 3: Training and Evaluation (pending)

## License

This project is part of a Capstone assignment. Dataset provided by BraTS 2024.

## References

- BraTS 2024 Dataset: https://www.med.upenn.edu/cbica/brats2024/
- PyTorch: https://pytorch.org/
- Nibabel: https://nipy.org/nibabel/

---

**Status**: ✅ Preprocessing Pipeline Complete and Tested
**Last Updated**: May 4, 2026
