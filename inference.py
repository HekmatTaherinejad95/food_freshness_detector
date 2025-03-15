"""
Real-time inference script for the Food Freshness Detector model.

This module provides functionality for real-time food freshness classification
using a trained model, with support for both image-only and multi-modal approaches.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model import get_model


class FoodFreshnessDetector:
    """Class for real-time food freshness detection."""
    
    def __init__(self, 
                model_path: str,
                model_type: str = 'image_only',
                base_model: str = 'resnet18',
                num_classes: int = 3,
                class_names: Optional[List[str]] = None,
                device: str = 'cuda',
                image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the food freshness detector.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('image_only' or 'multi_modal')
            base_model: Base model architecture
            num_classes: Number of output classes
            class_names: List of class names (if None, uses default names)
            device: Device to run inference on ('cuda' or 'cpu')
            image_size: Size to resize input images to
        """
        self.model_type = model_type
        self.image_size = image_size
        
        # Set default class names if not provided
        if class_names is None:
            self.class_names = ['Fresh', 'Ripe', 'Spoiled']
        else:
            self.class_names = class_names
        
        # Initialize device
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Create model
        self.model = get_model(
            model_type=model_type,
            num_classes=num_classes,
            base_model=base_model,
            pretrained=False  # We'll load weights from checkpoint
        )
        
        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model. Results may not be accurate.")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert numpy array to PIL Image
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def read_sensor_data(self, sensor_values: Optional[List[float]] = None) -> torch.Tensor:
        """
        Process sensor data for multi-modal inference.
        
        Args:
            sensor_values: List of sensor readings (if None, uses random values for demonstration)
            
        Returns:
            Sensor data tensor
        """
        if sensor_values is None:
            # Generate random sensor values for demonstration
            sensor_values = [np.random.uniform(0, 1) for _ in range(3)]
            print(f"Using random sensor values: {sensor_values}")
        
        # Convert to tensor
        sensor_tensor = torch.tensor([sensor_values], dtype=torch.float32)
        
        return sensor_tensor
    
    def predict(self, 
               image: Union[str, np.ndarray, Image.Image],
               sensor_values: Optional[List[float]] = None) -> Dict[str, Union[int, float, List[float]]]:
        """
        Predict the freshness of a food item.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            sensor_values: List of sensor readings (required for multi-modal model)
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            if self.model_type == 'image_only':
                outputs = self.model(image_tensor)
            else:
                # For multi-modal model, we need sensor data
                sensor_tensor = self.read_sensor_data(sensor_values).to(self.device)
                outputs = self.model(image_tensor, sensor_tensor)
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            # Convert all probabilities to list
            all_probs = probabilities.cpu().numpy().tolist()
        
        # Return results
        return {
            'class_id': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': all_probs
        }
    
    def visualize_prediction(self, 
                           image: Union[str, np.ndarray, Image.Image],
                           prediction: Dict[str, Union[int, float, List[float]]],
                           save_path: Optional[str] = None) -> None:
        """
        Visualize the prediction results.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            prediction: Prediction results from predict()
            save_path: Path to save the visualization (optional)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert PIL Image to numpy array
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Prediction: {prediction['class_name']}")
        plt.axis('off')
        
        # Plot probabilities
        plt.subplot(1, 2, 2)
        bars = plt.bar(self.class_names, prediction['probabilities'])
        
        # Color the predicted class
        bars[prediction['class_id']].set_color('red')
        
        plt.title(f"Confidence: {prediction['confidence']:.2f}")
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        
        plt.show()


class RealTimeDetector:
    """Class for real-time food freshness detection using a webcam."""
    
    def __init__(self, detector: FoodFreshnessDetector):
        """
        Initialize the real-time detector.
        
        Args:
            detector: FoodFreshnessDetector instance
        """
        self.detector = detector
    
    def start_webcam(self, camera_id: int = 0, quit_key: str = 'q') -> None:
        """
        Start real-time detection using webcam.
        
        Args:
            camera_id: Camera device ID
            quit_key: Key to press to quit
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print(f"Starting real-time detection. Press '{quit_key}' to quit.")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Make prediction
            start_time = time.time()
            prediction = self.detector.predict(frame)
            inference_time = time.time() - start_time
            
            # Display result on frame
            class_name = prediction['class_name']
            confidence = prediction['confidence']
            
            # Choose color based on class (green for Fresh, yellow for Ripe, red for Spoiled)
            if class_name == 'Fresh':
                color = (0, 255, 0)  # Green in BGR
            elif class_name == 'Ripe':
                color = (0, 255, 255)  # Yellow in BGR
            else:
                color = (0, 0, 255)  # Red in BGR
            
            # Add text to frame
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Inference time: {inference_time*1000:.1f} ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Food Freshness Detector', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function for real-time food freshness detection."""
    parser = argparse.ArgumentParser(description='Real-time Food Freshness Detection')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='image_only',
                        choices=['image_only', 'multi_modal'],
                        help='Type of model to use')
    parser.add_argument('--base_model', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'mobilenet_v2'],
                        help='Base model architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if not using webcam)')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for real-time detection')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID for webcam')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create detector
    detector = FoodFreshnessDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        base_model=args.base_model,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Run inference
    if args.webcam:
        # Real-time detection using webcam
        real_time_detector = RealTimeDetector(detector)
        real_time_detector.start_webcam(camera_id=args.camera_id)
    elif args.image:
        # Single image inference
        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} not found")
            return
        
        # Make prediction
        prediction = detector.predict(args.image)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Class: {prediction['class_name']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        
        # Visualize prediction
        save_path = output_dir / f"prediction_{Path(args.image).stem}.png"
        detector.visualize_prediction(args.image, prediction, save_path=str(save_path))
    else:
        print("Error: Either --image or --webcam must be specified")


if __name__ == "__main__":
    main()
