"""
Data loading utilities for the Food Freshness Detector project.

This module provides PyTorch dataset and dataloader classes for loading
and batching food freshness images for model training and evaluation.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FoodFreshnessDataset(Dataset):
    """PyTorch Dataset for food freshness classification."""
    
    def __init__(self, 
                 data_dir: str, 
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            transform: Optional transform to apply to images
            target_size: Size to resize images to
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        
        # Find all image files and their labels
        self.samples = []
        self.class_to_idx = {}
        self._find_classes()
        self._make_dataset()
        
    def _find_classes(self) -> None:
        """Find class names and create class-to-index mapping."""
        classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
        
    def _make_dataset(self) -> None:
        """Create a list of (image_path, class_index) tuples."""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_file in class_dir.glob('*.*'):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((str(img_file), class_idx))
                    
        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
                
            return img, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder in case of error
            placeholder = torch.zeros(3, *self.target_size)
            return placeholder, label


def get_data_transforms(target_size: Tuple[int, int] = (224, 224)) -> Dict[str, transforms.Compose]:
    """
    Get data transforms for training, validation, and testing.
    
    Args:
        target_size: Size to resize images to
        
    Returns:
        Dictionary of transforms for each split
    """
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((target_size[0] + 20, target_size[1] + 20)),
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms


def create_dataloaders(data_dir: str, 
                      batch_size: int = 32, 
                      num_workers: int = 4,
                      target_size: Tuple[int, int] = (224, 224)) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Base directory containing train, val, and test subdirectories
        batch_size: Batch size for training and evaluation
        num_workers: Number of worker processes for data loading
        target_size: Size to resize images to
        
    Returns:
        Dictionary of DataLoaders for each split
    """
    data_dir = Path(data_dir)
    
    # Get data transforms
    data_transforms = get_data_transforms(target_size)
    
    # Create datasets
    image_datasets = {
        split: FoodFreshnessDataset(
            data_dir=data_dir / split,
            transform=data_transforms[split],
            target_size=target_size
        )
        for split in ['train', 'val', 'test'] if (data_dir / split).exists()
    }
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        for split, dataset in image_datasets.items()
    }
    
    # Get dataset sizes and class names
    dataset_sizes = {split: len(dataset) for split, dataset in image_datasets.items()}
    class_names = image_datasets['train'].classes if 'train' in image_datasets else []
    
    print(f"Created dataloaders with {len(class_names)} classes")
    for split, size in dataset_sizes.items():
        print(f"  {split}: {size} images")
    
    return dataloaders


def main():
    """Test the data loading functionality."""
    # Create dataloaders
    data_dir = 'processed_data'
    dataloaders = create_dataloaders(data_dir, batch_size=16, num_workers=2)
    
    # Test iterating through a batch
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i}: inputs shape {inputs.shape}, labels shape {labels.shape}")
            if i >= 2:  # Just test a few batches
                break
    else:
        print("No training data found. Please prepare the dataset first.")


if __name__ == "__main__":
    main()
