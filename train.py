"""
Training script for the Food Freshness Detector model.

This module implements the training pipeline for the food freshness classification model,
including data loading, model training, validation, and checkpointing.
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import create_dataloaders
from model import get_model, get_optimizer, get_loss_function


def train_model(model: nn.Module,
               dataloaders: Dict[str, DataLoader],
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
               num_epochs: int = 25,
               device: str = 'cuda',
               checkpoint_dir: str = 'checkpoints') -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model.
    
    Args:
        model: Model to train
        dataloaders: Dictionary of dataloaders for training and validation
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Trained model and dictionary of training history
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Initialize best model tracking
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Start training
    print(f"Starting training on {device}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                if scheduler is not None:
                    scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}_acc_{epoch_acc:.4f}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        print()
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_accuracy': best_acc,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return model, history


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train Food Freshness Detector model')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Directory containing the processed dataset')
    parser.add_argument('--model_type', type=str, default='image_only',
                        choices=['image_only', 'multi_modal'],
                        help='Type of model to train')
    parser.add_argument('--base_model', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'mobilenet_v2'],
                        help='Base model architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pre-trained weights')
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze base model weights')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Create model
    model = get_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        base_model=args.base_model,
        pretrained=not args.no_pretrained,
        freeze_base=args.freeze_base
    )
    
    # Create optimizer and loss function
    optimizer = get_optimizer(
        model=model,
        optimizer_name='adam',
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=7,
        gamma=0.1
    )
    
    # Create loss function
    criterion = get_loss_function(loss_name='cross_entropy')
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Plot training history
    plot_training_history(
        history=history,
        save_path=os.path.join(args.checkpoint_dir, 'training_history.png')
    )


if __name__ == "__main__":
    main()
