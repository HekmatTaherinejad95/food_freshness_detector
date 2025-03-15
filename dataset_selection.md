# Dataset Selection for Food Freshness Detector

After researching available datasets for food freshness classification, I've identified three promising candidates:

## 1. Fresh and Rotten Classification Dataset (Kaggle)
- **URL**: https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification
- **Description**: High-quality images of various fruits and vegetables in both fresh and rotten/stale states
- **Features**: Diverse lighting conditions, angles, and backgrounds; clear distinction between fresh and rotten states
- **Classes**: Binary classification (fresh vs. rotten)
- **Advantages**: High-quality images, diverse conditions mimicking real-world scenarios
- **Limitations**: Only binary classification (no "ripe" or "medium-fresh" category)

## 2. Fruits and Vegetables Dataset (Kaggle)
- **URL**: https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset
- **Description**: Contains images of fresh and rotten fruits and vegetables
- **Size**: ~12,000 images (5,997 fruit images for 10 classes, 6,003 vegetable images for 10 classes)
- **Classes**: 20 classes total (5 fresh fruits, 5 rotten fruits, 5 fresh vegetables, 5 rotten vegetables)
- **Advantages**: Good balance between fruits and vegetables, substantial number of images
- **Limitations**: Binary classification per item (fresh vs. rotten)

## 3. FruitVeg Freshness Dataset
- **URL**: Referenced in research paper: https://www.sciencedirect.com/org/science/article/pii/S1546221822007160
- **Description**: Comprehensive dataset specifically designed for freshness categorization
- **Size**: 60K images of 11 fruits and vegetables
- **Classes**: Three-level classification (pure-fresh, medium-fresh, rotten) for each fruit/vegetable
- **Advantages**: Includes the medium-fresh category, which aligns perfectly with our three-class requirement
- **Limitations**: May require more processing power due to larger size

## Selected Dataset

For the Food Freshness Detector project, I recommend using the **FruitVeg Freshness Dataset** for the following reasons:

1. It provides three freshness categories (pure-fresh, medium-fresh, rotten) which directly aligns with our project requirements for classifying items as "fresh," "ripe," or "spoiled"
2. With 60K images, it offers a substantial amount of training data
3. It covers 11 different fruits and vegetables, providing good variety
4. It was specifically designed for freshness categorization research

However, since this dataset may not be directly downloadable, we should implement a fallback plan:

### Primary Plan:
Attempt to locate and download the FruitVeg Freshness Dataset from the paper authors or associated repositories.

### Fallback Plan:
If the FruitVeg Freshness Dataset is not accessible, we will use the "Fruits and Vegetables Dataset" from Kaggle and adapt it to our needs:
1. Download the dataset with its binary classification (fresh/rotten)
2. Implement data augmentation to create a synthetic "ripe" category
3. Use color and texture analysis to identify items in the middle stage between fresh and rotten

## Dataset Preparation Strategy

Regardless of which dataset we use, we will implement the following preparation steps:

1. **Dataset Organization**:
   - Split into training (70%), validation (15%), and test (15%) sets
   - Ensure balanced class distribution in each split

2. **Data Preprocessing**:
   - Resize images to a standard size (e.g., 224×224 pixels)
   - Normalize pixel values
   - Apply color correction if needed

3. **Data Augmentation**:
   - Random rotations (±30 degrees)
   - Random horizontal and vertical flips
   - Random brightness and contrast adjustments
   - Random cropping
   - Slight color jitter to simulate different lighting conditions

This preparation strategy will help ensure our model is robust and generalizes well to new, unseen images of fruits and vegetables in various freshness states.
