"""
Tumor Classification using EfficientNet

This script implements a baseline tumor typing model using EfficientNet-B4
for classifying glioma tumors (LGG vs HGG) from 2D FLAIR MRI slices.

Features:
- EfficientNet-B4 with ImageNet pre-trained weights
- Transfer learning with fine-tuned classifier
- Training on BraTS dataset patient slices
- Inference function for cropped tumor images

Usage:
- Run the script to train the model: python classification.py
- The trained model will be saved as 'efficientnet_tumor_classifier.pth'
- Use the predict() function for inference on new images
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from typing import Tuple


class TumorClassificationDataset(Dataset):
    """
    Dataset for tumor classification from 2D FLAIR slices.
    
    Each patient provides one middle slice labeled by glioma grade:
    - 100: Low Grade Glioma (LGG) -> class 0
    - 101+: High Grade Glioma (HGG) -> class 1
    """
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.patients = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
        self.transform = transform
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        case_path = os.path.join(self.data_dir, patient)
        
        # Load FLAIR modality
        flair_files = (glob.glob(os.path.join(case_path, '*-t2f.nii.gz')) or 
                      glob.glob(os.path.join(case_path, '*-flair.nii.gz')))
        
        if not flair_files:
            raise FileNotFoundError(f"Missing FLAIR file in {case_path}")
        
        img = nib.load(flair_files[0]).get_fdata().astype(np.float32)
        
        # Take middle axial slice
        middle_slice = img.shape[2] // 2
        slice_img = img[:, :, middle_slice]
        
        # Normalize to [0, 1]
        slice_min, slice_max = slice_img.min(), slice_img.max()
        if slice_max > slice_min:
            slice_img = (slice_img - slice_min) / (slice_max - slice_min)
        else:
            slice_img = np.zeros_like(slice_img)
        
        # Convert to 3-channel RGB (repeat grayscale)
        slice_img = np.stack([slice_img] * 3, axis=-1)  # (H, W, 3)
        
        # Convert to tensor and permute to (C, H, W)
        slice_img = torch.tensor(slice_img, dtype=torch.float32).permute(2, 0, 1)
        
        # Extract grade from patient folder name
        grade = int(patient.split('-')[-1])
        label = 0 if grade == 100 else 1  # 0: LGG, 1: HGG
        
        if self.transform:
            slice_img = self.transform(slice_img)
            
        return slice_img, label


def create_model(num_classes: int = 2) -> nn.Module:
    """
    Create EfficientNet-B4 model with transfer learning.
    
    Args:
        num_classes: Number of output classes (default: 2 for LGG/HGG)
    
    Returns:
        nn.Module: Modified EfficientNet-B4 model
    """
    # Load pre-trained EfficientNet-B4
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_features = model.classifier[1].in_features  # 1792 for B4
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, num_epochs: int = 10, 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the classification model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
    """
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'efficientnet_tumor_classifier.pth')
            print(f'  Model saved with accuracy: {best_accuracy:.2f}%')


def predict_tumor_type(image: np.ndarray, model_path: str = 'efficientnet_tumor_classifier.pth',
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[np.ndarray, str]:
    """
    Predict tumor type from a cropped tumor image.
    
    Args:
        image: Cropped tumor image as numpy array. 
               Can be (H, W) grayscale or (H, W, 3) RGB
        model_path: Path to saved model weights
        device: Device for inference
    
    Returns:
        Tuple of (probabilities, predicted_class_name)
        probabilities: np.ndarray of shape (num_classes,) with softmax probabilities
        predicted_class_name: 'LGG' or 'HGG'
    """
    # Load model
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess image
    if image.ndim == 2:
        # Grayscale to RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        # Single channel to RGB
        image = np.repeat(image, 3, axis=-1)
    
    # Normalize to [0, 1] if not already
    image_min, image_max = image.min(), image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    
    # Convert to tensor (C, H, W)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    
    # Class names
    class_names = ['LGG', 'HGG']
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    
    return probabilities, predicted_class


def main():
    """Main training function."""
    data_dir = 'data'
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        return
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet input size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TumorClassificationDataset(data_dir, transform=transform)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create model
    model = create_model()
    
    # Train
    train_model(model, train_loader, val_loader, num_epochs=10)
    
    print("Training completed! Model saved as 'efficientnet_tumor_classifier.pth'")


if __name__ == '__main__':
    main()