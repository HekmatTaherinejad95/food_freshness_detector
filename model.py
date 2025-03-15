"""
Model architecture for the Food Freshness Detector project.

This module defines the CNN architecture for classifying food freshness
based on images. It includes both a standard image-only model and an
optional multi-modal approach that can incorporate gas sensor data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Tuple, Union


class FoodFreshnessClassifier(nn.Module):
    """
    CNN model for food freshness classification based on pre-trained architectures.
    
    This model uses transfer learning with pre-trained models like ResNet18
    or EfficientNet and fine-tunes them for food freshness classification.
    """
    
    def __init__(self, 
                 num_classes: int = 3, 
                 base_model: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_base: bool = False):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes (e.g., fresh, ripe, spoiled)
            base_model: Name of the pre-trained model to use as base
            pretrained: Whether to use pre-trained weights
            freeze_base: Whether to freeze the base model weights
        """
        super(FoodFreshnessClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.base_model_name = base_model
        
        # Initialize the base model
        if base_model == 'resnet18':
            self.base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
            
        elif base_model == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()  # Remove the classifier
            
        elif base_model == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()  # Remove the classifier
            
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features from the base model
        features = self.base_model(x)
        
        # Pass through the classifier
        output = self.classifier(features)
        
        return output
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the feature embedding for an input image.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature embedding tensor
        """
        return self.base_model(x)


class MultiModalFoodFreshnessClassifier(nn.Module):
    """
    Multi-modal model for food freshness classification using both image and sensor data.
    
    This model combines a CNN for image processing with a neural network for
    processing gas sensor data, using a fusion layer to integrate both inputs.
    """
    
    def __init__(self, 
                 num_classes: int = 3, 
                 base_model: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_base: bool = False,
                 num_sensors: int = 1):
        """
        Initialize the multi-modal model.
        
        Args:
            num_classes: Number of output classes (e.g., fresh, ripe, spoiled)
            base_model: Name of the pre-trained model to use as base for image processing
            pretrained: Whether to use pre-trained weights
            freeze_base: Whether to freeze the base model weights
            num_sensors: Number of gas sensor inputs
        """
        super(MultiModalFoodFreshnessClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.num_sensors = num_sensors
        
        # Image processing branch
        self.image_model = FoodFreshnessClassifier(
            num_classes=num_classes,
            base_model=base_model,
            pretrained=pretrained,
            freeze_base=freeze_base
        )
        
        # Get the feature dimension from the image model
        if base_model == 'resnet18':
            image_features = 512
        elif base_model == 'efficientnet_b0':
            image_features = 1280
        elif base_model == 'mobilenet_v2':
            image_features = 1280
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Sensor processing branch
        self.sensor_processor = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(image_features + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-modal model.
        
        Args:
            image: Input image tensor of shape (batch_size, channels, height, width)
            sensor_data: Input sensor data tensor of shape (batch_size, num_sensors)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Process image
        image_features = self.image_model.get_embedding(image)
        
        # Process sensor data
        sensor_features = self.sensor_processor(sensor_data)
        
        # Concatenate features
        combined_features = torch.cat((image_features, sensor_features), dim=1)
        
        # Fusion and classification
        output = self.fusion(combined_features)
        
        return output


def get_model(model_type: str = 'image_only', 
             num_classes: int = 3, 
             base_model: str = 'resnet18',
             pretrained: bool = True,
             freeze_base: bool = False,
             num_sensors: int = 1) -> nn.Module:
    """
    Factory function to create a model based on specified parameters.
    
    Args:
        model_type: Type of model ('image_only' or 'multi_modal')
        num_classes: Number of output classes
        base_model: Name of the pre-trained model to use as base
        pretrained: Whether to use pre-trained weights
        freeze_base: Whether to freeze the base model weights
        num_sensors: Number of gas sensor inputs (for multi_modal only)
        
    Returns:
        Initialized model
    """
    if model_type == 'image_only':
        return FoodFreshnessClassifier(
            num_classes=num_classes,
            base_model=base_model,
            pretrained=pretrained,
            freeze_base=freeze_base
        )
    elif model_type == 'multi_modal':
        return MultiModalFoodFreshnessClassifier(
            num_classes=num_classes,
            base_model=base_model,
            pretrained=pretrained,
            freeze_base=freeze_base,
            num_sensors=num_sensors
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_optimizer(model: nn.Module, 
                 optimizer_name: str = 'adam',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer ('adam', 'sgd', etc.)
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 penalty)
        
    Returns:
        Initialized optimizer
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_loss_function(loss_name: str = 'cross_entropy',
                     class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Create a loss function.
    
    Args:
        loss_name: Name of the loss function
        class_weights: Optional tensor of class weights for weighted loss
        
    Returns:
        Loss function
    """
    if loss_name.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name.lower() == 'focal':
        # Focal loss for imbalanced datasets
        from torch.nn import functional as F
        
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None):
                super(FocalLoss, self).__init__()
                self.gamma = gamma
                self.alpha = alpha
                
            def forward(self, input, target):
                ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha=class_weights)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def main():
    """Test the model architecture."""
    # Create a sample input
    batch_size = 4
    channels = 3
    height, width = 224, 224
    num_classes = 3
    
    # Create a sample image tensor
    image = torch.randn(batch_size, channels, height, width)
    
    # Test image-only model
    print("Testing image-only model...")
    model = get_model(model_type='image_only', num_classes=num_classes)
    output = model(image)
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model architecture:\n{model}")
    
    # Test multi-modal model
    print("\nTesting multi-modal model...")
    num_sensors = 3
    sensor_data = torch.randn(batch_size, num_sensors)
    multi_modal_model = get_model(
        model_type='multi_modal',
        num_classes=num_classes,
        num_sensors=num_sensors
    )
    output = multi_modal_model(image, sensor_data)
    print(f"Image input shape: {image.shape}")
    print(f"Sensor input shape: {sensor_data.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test optimizer and loss function
    optimizer = get_optimizer(model)
    loss_fn = get_loss_function()
    print(f"\nOptimizer: {optimizer}")
    print(f"Loss function: {loss_fn}")


if __name__ == "__main__":
    main()
