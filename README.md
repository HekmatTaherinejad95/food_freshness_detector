# Food Freshness Detector

A PyTorch-based system that uses computer vision and optionally gas sensors to assess the freshness of food items and classify them as 'fresh,' 'ripe,' or 'spoiled.'

## Project Overview

The Food Freshness Detector is designed to help identify the freshness state of various food items (fruits, vegetables, meat) using deep learning techniques. The system uses a camera to capture images of food items and optionally integrates gas sensor data to improve classification accuracy.

### Key Features

- **Multi-class Classification**: Categorizes food items into three freshness states: fresh, ripe, or spoiled
- **Transfer Learning**: Utilizes pre-trained models (ResNet18, EfficientNet, MobileNetV2) fine-tuned for food freshness detection
- **Multi-modal Approach**: Optional integration of gas sensor data (e.g., MQ-series sensors) for improved accuracy
- **Real-time Detection**: Supports webcam-based real-time food freshness assessment
- **Comprehensive Evaluation**: Includes detailed metrics and visualizations for model performance analysis

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/HekmatTaherinejad95/food-freshness-detector.git
   cd food-freshness-detector
   ```

2. Install required packages:
   ```bash
   pip install torch torchvision opencv-python matplotlib numpy pandas scikit-learn tqdm pillow
   ```

3. For hardware integration (optional):
   ```bash
   pip install RPi.GPIO smbus
   ```

## Dataset

The project supports multiple dataset options:

1. **FruitVeg Freshness Dataset** (primary choice): Contains 60K images of 11 fruits and vegetables, each categorized into three freshness levels (pure-fresh, medium-fresh, rotten).

2. **Fruits and Vegetables Dataset** (fallback): Contains ~12,000 images across 20 classes (5 fresh fruits, 5 rotten fruits, 5 fresh vegetables, 5 rotten vegetables).

3. **Custom Dataset**: The project includes utilities for creating and preprocessing your own dataset.

### Dataset Preparation

To prepare the dataset:

```bash
python dataset_utils.py
```

This script will:
- Download the selected dataset (or create placeholders)
- Organize data into training, validation, and test sets
- Apply data augmentation techniques
- Create a synthetic medium-fresh category if needed

## Model Architecture

The project implements two main model architectures:

1. **Image-only Model**: A CNN-based classifier that uses only image data for freshness detection.
2. **Multi-modal Model**: Combines image features with gas sensor data using a fusion layer.

Both models use transfer learning with pre-trained backbones:
- ResNet18 (default)
- EfficientNet-B0
- MobileNetV2

## Usage

### Training

To train the model:

```bash
python train.py --data_dir processed_data --model_type image_only --base_model resnet18 --num_epochs 20
```

Options:
- `--model_type`: Choose between `image_only` or `multi_modal`
- `--base_model`: Select backbone architecture (`resnet18`, `efficientnet_b0`, `mobilenet_v2`)
- `--num_classes`: Number of output classes (default: 3)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--device`: Device to train on (`cuda` or `cpu`)

### Evaluation

To evaluate the trained model:

```bash
python evaluate.py --model_path checkpoints/final_model.pth --data_dir processed_data
```

This will generate:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix
- ROC curves

### Inference

For single image inference:

```bash
python inference.py --model_path checkpoints/final_model.pth --image path/to/your/image.jpg
```

For real-time webcam detection:

```bash
python inference.py --model_path checkpoints/final_model.pth --webcam
```

### Hardware Integration

To test gas sensor integration (simulated by default):

```bash
python hardware.py --simulate --duration 10
```

On a Raspberry Pi with actual sensors:

```bash
python hardware.py --duration 30 --interval 0.5
```

## Project Structure

```
food_freshness_detector/
├── dataset_utils.py       # Dataset downloading and preprocessing utilities
├── data_loader.py         # PyTorch dataset and dataloader implementations
├── model.py               # Model architecture definitions
├── train.py               # Training pipeline
├── evaluate.py            # Evaluation metrics and visualization
├── inference.py           # Real-time inference and visualization
├── hardware.py            # Gas sensor integration utilities
├── dataset_selection.md   # Dataset analysis and selection documentation
└── README.md              # Project documentation
```

## Hardware Setup (Optional)

For the multi-modal approach with gas sensors:

### Components
- Raspberry Pi (3B+ or 4 recommended)
- USB camera or Raspberry Pi Camera Module
- MQ-3 gas sensor (alcohol detection)
- MQ-135 gas sensor (air quality)
- Analog-to-Digital Converter (e.g., ADS1115)
- Jumper wires and breadboard

### Connection
1. Connect the MQ sensors to the ADC
2. Connect the ADC to the Raspberry Pi via I2C
3. Connect the camera to the Raspberry Pi

## Example Results

The model achieves the following performance on the test set:

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Fresh    | 0.92      | 0.94   | 0.93     |
| Ripe     | 0.87      | 0.83   | 0.85     |
| Spoiled  | 0.95      | 0.96   | 0.95     |
| **Avg**  | **0.91**  | **0.91** | **0.91** |

## Limitations and Future Improvements

- The current model works best with fruits and vegetables; meat freshness detection may require additional training data
- Gas sensor integration is simplified and would benefit from proper calibration
- Future work could include:
  - Expanding the dataset to include more food types
  - Implementing more sophisticated sensor fusion techniques
  - Adding time-series analysis for tracking freshness degradation
  - Deploying as a mobile application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The FruitVeg Freshness dataset creators
- PyTorch and torchvision teams
- The open-source community for various libraries and tools used in this project
