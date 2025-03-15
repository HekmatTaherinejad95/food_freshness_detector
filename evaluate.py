"""
Evaluation script for the Food Freshness Detector model.

This module implements evaluation metrics and functions to assess the 
performance of the food freshness classification model on test data.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from data_loader import create_dataloaders
from model import get_model


def evaluate_model(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: str = 'cuda') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (accuracy, true labels, predicted labels, prediction probabilities)
    """
    # Initialize device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    all_labels = []
    all_preds = []
    all_probs = []
    correct = 0
    total = 0
    
    # Evaluate model
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    
    return accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(true_labels: np.ndarray, 
                         pred_labels: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        true_labels: Array of true labels
        pred_labels: Array of predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(true_labels: np.ndarray, 
                  pred_probs: np.ndarray, 
                  class_names: List[str],
                  save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        true_labels: Array of true labels
        pred_probs: Array of prediction probabilities
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize the labels
    n_classes = len(class_names)
    true_labels_bin = label_binarize(true_labels, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, color, cls in zip(range(n_classes), colors[:n_classes], class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {cls} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved ROC curve to {save_path}")
    
    plt.show()


def print_classification_metrics(true_labels: np.ndarray, 
                               pred_labels: np.ndarray, 
                               class_names: List[str]) -> None:
    """
    Print classification metrics.
    
    Args:
        true_labels: Array of true labels
        pred_labels: Array of predicted labels
        class_names: List of class names
    """
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    
    print("\nOverall Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


def main():
    """Main function to evaluate the model."""
    parser = argparse.ArgumentParser(description='Evaluate Food Freshness Detector model')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Directory containing the processed dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='image_only',
                        choices=['image_only', 'multi_modal'],
                        help='Type of model to evaluate')
    parser.add_argument('--base_model', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'mobilenet_v2'],
                        help='Base model architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to evaluate on')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Get test dataloader
    test_dataloader = dataloaders.get('test')
    if test_dataloader is None:
        print("Error: Test dataloader not found. Make sure the test data is available.")
        return
    
    # Get class names
    class_names = test_dataloader.dataset.classes
    
    # Create model
    model = get_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        base_model=args.base_model,
        pretrained=False  # We'll load weights from checkpoint
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    accuracy, true_labels, pred_labels, pred_probs = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=args.device
    )
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Print classification metrics
    print_classification_metrics(true_labels, pred_labels, class_names)
    
    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(true_labels, pred_labels, class_names, save_path=str(cm_path))
    
    # Plot ROC curve
    roc_path = output_dir / 'roc_curve.png'
    plot_roc_curve(true_labels, pred_probs, class_names, save_path=str(roc_path))
    
    print(f"\nEvaluation results saved to {output_dir}")


if __name__ == "__main__":
    main()
