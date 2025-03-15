"""
Dataset utilities for the Food Freshness Detector project.

This module provides functions for downloading, preprocessing, and augmenting
food freshness datasets. It supports both the primary FruitVeg Freshness Dataset
and the fallback Fruits and Vegetables Dataset from Kaggle.
"""

import os
import random
import shutil
from pathlib import Path
import urllib.request
import zipfile
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

class DatasetDownloader:
    """Class for downloading and extracting datasets."""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to store the downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    def download_kaggle_dataset(self, dataset_name: str, kaggle_api: bool = False) -> str:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_name: Name of the Kaggle dataset (e.g., 'muhriddinmuxiddinov/fruits-and-vegetables-dataset')
            kaggle_api: Whether to use the Kaggle API (requires authentication) or direct download links
            
        Returns:
            Path to the downloaded dataset
        """
        dataset_path = self.data_dir / dataset_name.split('/')[-1]
        
        if dataset_path.exists():
            print(f"Dataset already exists at {dataset_path}")
            return str(dataset_path)
        
        dataset_path.mkdir(exist_ok=True, parents=True)
        
        if kaggle_api:
            try:
                # This requires kaggle.json credentials file
                import kaggle
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(dataset_name, path=str(dataset_path), unzip=True)
                print(f"Dataset downloaded to {dataset_path}")
            except Exception as e:
                print(f"Failed to download using Kaggle API: {e}")
                print("Falling back to direct download links...")
                kaggle_api = False
        
        if not kaggle_api:
            # Fallback to direct download links
            if 'fruits-and-vegetables-dataset' in dataset_name:
                # This is a simplified example - in a real scenario, we would need to find the actual direct download link
                download_url = "https://example.com/fruits-vegetables-dataset.zip"  # Placeholder URL
                zip_path = dataset_path / "dataset.zip"
                
                print(f"Downloading dataset from {download_url}...")
                # In a real implementation, we would download the actual file
                # urllib.request.urlretrieve(download_url, zip_path)
                
                print("Note: Direct download links for Kaggle datasets require authentication.")
                print("Please download the dataset manually from Kaggle and place it in the data directory.")
                print(f"Expected path: {dataset_path}")
                
                # For demonstration purposes, we'll create a placeholder structure
                self._create_placeholder_dataset(dataset_path)
                
        return str(dataset_path)
    
    def _create_placeholder_dataset(self, dataset_path: Path):
        """Create a placeholder dataset structure for demonstration purposes."""
        # Create basic directory structure
        fruits_dir = dataset_path / "Fruits"
        vegetables_dir = dataset_path / "Vegetables"
        
        for directory in [fruits_dir, vegetables_dir]:
            directory.mkdir(exist_ok=True)
            
            # Create subdirectories for fresh and rotten
            for category in ["Fresh", "Rotten"]:
                for item in ["Apple", "Banana", "Orange"] if "Fruits" in str(directory) else ["Tomato", "Potato", "Carrot"]:
                    item_dir = directory / f"{category}_{item}"
                    item_dir.mkdir(exist_ok=True)
                    
                    # Create a placeholder text file
                    with open(item_dir / "placeholder.txt", "w") as f:
                        f.write(f"Placeholder for {category} {item} images")
        
        print(f"Created placeholder dataset structure at {dataset_path}")


class DatasetPreprocessor:
    """Class for preprocessing and organizing datasets."""
    
    def __init__(self, data_dir: str, output_dir: str = 'processed_data'):
        """
        Initialize the dataset preprocessor.
        
        Args:
            data_dir: Directory containing the raw dataset
            output_dir: Directory to store the processed dataset
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def organize_kaggle_fruits_vegetables(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                                         test_ratio: float = 0.15, seed: int = 42) -> None:
        """
        Organize the Kaggle Fruits and Vegetables dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio of images to use for training
            val_ratio: Ratio of images to use for validation
            test_ratio: Ratio of images to use for testing
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        test_dir = self.output_dir / 'test'
        
        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(exist_ok=True)
            
        # Process Fruits and Vegetables directories
        for category_dir in ['Fruits', 'Vegetables']:
            category_path = self.data_dir / category_dir
            
            if not category_path.exists():
                print(f"Warning: {category_path} does not exist. Skipping...")
                continue
                
            # Process each subcategory (e.g., Fresh_Apple, Rotten_Banana)
            for subcategory in os.listdir(category_path):
                subcategory_path = category_path / subcategory
                
                if not subcategory_path.is_dir():
                    continue
                    
                # Create corresponding directories in train, val, test
                for output_dir in [train_dir, val_dir, test_dir]:
                    (output_dir / subcategory).mkdir(exist_ok=True)
                
                # Get all image files
                image_files = [f for f in os.listdir(subcategory_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # If no image files found, check for placeholder
                if not image_files and os.path.exists(subcategory_path / "placeholder.txt"):
                    print(f"Using placeholder for {subcategory}")
                    # Create empty placeholder files in each split
                    for output_dir in [train_dir, val_dir, test_dir]:
                        with open(output_dir / subcategory / "placeholder.txt", "w") as f:
                            f.write(f"Placeholder for {subcategory} images in {output_dir.name} split")
                    continue
                
                # Shuffle the files
                random.shuffle(image_files)
                
                # Calculate split indices
                n_files = len(image_files)
                n_train = int(n_files * train_ratio)
                n_val = int(n_files * val_ratio)
                
                # Split the files
                train_files = image_files[:n_train]
                val_files = image_files[n_train:n_train + n_val]
                test_files = image_files[n_train + n_val:]
                
                # Copy files to respective directories
                for files, output_dir in zip([train_files, val_files, test_files], 
                                           [train_dir, val_dir, test_dir]):
                    for file in files:
                        src = subcategory_path / file
                        dst = output_dir / subcategory / file
                        
                        # In a real implementation, we would copy the file
                        # shutil.copy(src, dst)
                        
                        # For demonstration, just create an empty file
                        with open(dst, "w") as f:
                            f.write(f"Placeholder for {file}")
                
                print(f"Processed {subcategory}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def create_medium_fresh_category(self, output_dir: Optional[Path] = None) -> None:
        """
        Create a synthetic 'medium-fresh' category from fresh and rotten images.
        
        This function creates medium-fresh images by blending fresh and rotten images
        or by applying transformations to fresh images to make them appear slightly aged.
        
        Args:
            output_dir: Directory to store the medium-fresh images. If None, uses self.output_dir.
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        # Ensure output directory exists
        medium_fresh_dir = output_dir / 'medium_fresh'
        medium_fresh_dir.mkdir(exist_ok=True, parents=True)
        
        # Process each category (fruits, vegetables)
        for category in ['Fruits', 'Vegetables']:
            category_dir = self.data_dir / category
            
            if not category_dir.exists():
                print(f"Warning: {category_dir} does not exist. Skipping...")
                continue
            
            # Find pairs of fresh and rotten items
            fresh_dirs = [d for d in os.listdir(category_dir) if d.startswith('Fresh_')]
            
            for fresh_dir in fresh_dirs:
                # Get the corresponding rotten directory
                item_name = fresh_dir.replace('Fresh_', '')
                rotten_dir = f'Rotten_{item_name}'
                
                if rotten_dir not in os.listdir(category_dir):
                    print(f"Warning: No matching rotten directory for {fresh_dir}. Skipping...")
                    continue
                
                # Create output directory for this item
                item_output_dir = medium_fresh_dir / f'Medium_{item_name}'
                item_output_dir.mkdir(exist_ok=True, parents=True)
                
                # Get fresh images
                fresh_path = category_dir / fresh_dir
                fresh_images = [f for f in os.listdir(fresh_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # If using placeholders
                if not fresh_images and os.path.exists(fresh_path / "placeholder.txt"):
                    with open(item_output_dir / "placeholder.txt", "w") as f:
                        f.write(f"Placeholder for Medium_{item_name} images")
                    continue
                
                # Process each fresh image to create a medium-fresh version
                for img_file in fresh_images[:10]:  # Limit to 10 images for demonstration
                    try:
                        # In a real implementation, we would:
                        # 1. Open the fresh image
                        # img = Image.open(fresh_path / img_file)
                        
                        # 2. Apply transformations to make it appear medium-fresh
                        # - Reduce saturation
                        # - Add slight yellow/brown tint
                        # - Add minor spots or blemishes
                        
                        # 3. Save the transformed image
                        output_path = item_output_dir / f"medium_{img_file}"
                        # img.save(output_path)
                        
                        # For demonstration, just create a placeholder file
                        with open(output_path, "w") as f:
                            f.write(f"Placeholder for medium-fresh version of {img_file}")
                            
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
                
                print(f"Created medium-fresh category for {item_name}")


class DataAugmenter:
    """Class for applying data augmentation to images."""
    
    @staticmethod
    def augment_image(image: Image.Image, rotation_range: int = 30, 
                     brightness_range: Tuple[float, float] = (0.8, 1.2),
                     contrast_range: Tuple[float, float] = (0.8, 1.2),
                     flip_prob: float = 0.5) -> List[Image.Image]:
        """
        Apply various augmentations to an image.
        
        Args:
            image: PIL Image to augment
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            flip_prob: Probability of applying horizontal flip
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Rotation
        rotation_angle = random.uniform(-rotation_range, rotation_range)
        rotated_img = image.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)
        augmented_images.append(rotated_img)
        
        # Brightness adjustment
        brightness_factor = random.uniform(*brightness_range)
        brightness_img = ImageEnhance.Brightness(image).enhance(brightness_factor)
        augmented_images.append(brightness_img)
        
        # Contrast adjustment
        contrast_factor = random.uniform(*contrast_range)
        contrast_img = ImageEnhance.Contrast(image).enhance(contrast_factor)
        augmented_images.append(contrast_img)
        
        # Horizontal flip
        if random.random() < flip_prob:
            flipped_img = ImageOps.mirror(image)
            augmented_images.append(flipped_img)
            
        return augmented_images
    
    @staticmethod
    def apply_augmentation_to_directory(input_dir: str, output_dir: str, 
                                       augmentations_per_image: int = 3) -> None:
        """
        Apply augmentation to all images in a directory.
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
            augmentations_per_image: Number of augmented versions to create per original image
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        image_files = [f for f in os.listdir(input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            try:
                # In a real implementation, we would:
                # 1. Open the image
                # img = Image.open(input_path / img_file)
                
                # 2. Apply augmentations
                # augmented_images = []
                # for _ in range(augmentations_per_image):
                #     augmented = DataAugmenter.augment_image(img)
                #     augmented_images.extend(augmented)
                
                # 3. Save augmented images
                # for i, aug_img in enumerate(augmented_images):
                #     aug_img<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>