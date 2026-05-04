
import nibabel as nib
import numpy as np
import os
import shutil
from pathlib import Path

def create_capstone_dummy_data(data_dir="working_data", n_patients=20, shape=(128, 128, 128)):
    data_dir = Path(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    mods = ["t1n", "t1c", "t2w", "t2f"]
    labels_in_data = [0, 2, 4]
    
    print(f"Generating {n_patients} synthetic patient cases...")
    for i in range(n_patients):
        pid = f"BraTS-GLI-{i:05d}-100"
        pdir = data_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        
        # Create modalities
        for mod in mods:
            # Using some patterns to make it slightly less random (simple sphere in center)
            data = np.random.rand(*shape).astype(np.float32) * 0.1
            xx, yy, zz = np.indices(shape)
            center = (shape[0]//2, shape[1]//2, shape[2]//2)
            radius = 20 + i # vary radius
            mask = (xx-center[0])**2 + (yy-center[1])**2 + (zz-center[2])**2 < radius**2
            data[mask] += 1.0 # Simulate tumor signal
            
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, str(pdir / f"{pid}-{mod}.nii.gz"))
            
        # Create segmentation (labels 0, 2, 4)
        seg_data = np.zeros(shape, dtype=np.float32)
        seg_data[mask] = 4 # ET
        # Add some core (label 2) inside the ET
        core_mask = (xx-center[0])**2 + (yy-center[1])**2 + (zz-center[2])**2 < (radius//2)**2
        seg_data[core_mask] = 2
        
        seg_img = nib.Nifti1Image(seg_data, np.eye(4))
        nib.save(seg_img, str(pdir / f"{pid}-seg.nii.gz"))
        
        if (i+1) % 5 == 0:
            print(f"  Progress: {i+1}/{n_patients} cases created.")

if __name__ == "__main__":
    create_capstone_dummy_data()
