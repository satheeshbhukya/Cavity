# Dental Cavity Detection and Segmentation System

## Overview
This project implements an advanced dental cavity detection and segmentation system using deep learning. The system combines YOLOv8x for object detection and UNet with an EfficientNet-B4 backbone for precise segmentation, addressing the challenging task of identifying dental cavities in X-ray images with high accuracy and detailed boundary delineation.

## Technical Architecture

### 1. Dataset Processing & Analysis
- **Input Structure**: Processes datasets containing dental X-rays (def-images/), YOLO format labels (def-labels/), and segmentation masks (mask_images/)
- **Dataset Analysis**: Validates image-label-mask correspondence and reports statistics
- **Data Preparation**: Implements train/validation split (80%/20%) with proper directory structure
- **Dataset YAML**: Generates configuration file with paths and class information for YOLO training

### 2. YOLOv8x Object Detection
- **Model**: Utilizes YOLOv8x architecture (largest variant) for maximum detection accuracy
- **Training Parameters**:
  - Batch size: 16 with gradient accumulation
  - Learning rate: 0.001 with cosine scheduling
  - Epochs: 30 with early stopping (patience=15)
  - Weight decay: 0.0005 for regularization
- **Advanced Augmentations**:
  - Mosaic (probability=1.0) for object localization improvement
  - Mixup (probability=0.1) for context learning
  - Copy-paste (probability=0.1) for instance variability
  - Rotation (±10°), translation (±10%), scaling (±50%), shear (±2°)
  - Horizontal flips (probability=0.5)
- **Precision Improvement**:
  - Custom filtering pipeline to address low precision (0.0036)
  - Size-based filtering (minimum area=400px)
  - Shape analysis (aspect ratio constraints: 0.4-2.5)
  - Edge detection rejection (30px margin)
  - Texture analysis (standard deviation threshold=15)
  - Context-aware filtering (comparing region brightness with surroundings)

### 3. UNet Segmentation Pipeline
- **Model Architecture**:
  - Encoder: EfficientNet-B4 pretrained on ImageNet
  - Decoder: UNet with skip connections
  - Output activation: Sigmoid for binary segmentation
  - Input resolution: 384×384 pixels (increased from standard 256×256)
- **Custom Dataset Implementation**:
  - DentalDataset class with LRU caching (cache_size=100)
  - Efficient loading with parallelization (num_workers=4)
  - Pin memory for faster GPU transfer
- **Advanced Data Augmentation**:
  - RandomResizedCrop with scale=(0.8, 1.0)
  - Elastic, grid, and optical distortions for dental-specific transformations
  - Noise and blur variations (Gaussian, median)
  - Intensity adjustments (CLAHE, brightness/contrast, gamma)
  - CoarseDropout to simulate occlusions
  - Normalization with ImageNet statistics
- **Loss Function**:
  - Custom DiceBCELoss combining binary cross-entropy and dice coefficient
  - Optimizes both pixel classification and region similarity
- **Training Optimizations**:
  - AdamW optimizer with weight decay=1e-5
  - Cosine annealing with warm restarts (T_0=10)
  - Mixed precision training with automatic gradient scaling
  - Gradient accumulation (steps=2) for effective batch size=32
  - Early stopping with validation monitoring

### 4. Combined Inference Pipeline
**CavityDetector Class**:
- Unified interface for both detection and segmentation
- Configurable confidence threshold (default=0.35)
- Segmentation mask threshold (default=0.45)
- Robust error handling with model loading verification

**Inference Process**:
- YOLO detection identifies potential cavity regions
- UNet generates precise segmentation masks
- Post-processing with morphological operations (closing, opening)
- Fallback mechanism to YOLO-based segmentation if UNet fails
- Visualization generation with bounding boxes and color-coded masks

### 6. Comprehensive Evaluation System
**DatasetEvaluator Class**:
- Parallel file validation using ThreadPoolExecutor
- Automatic matching of images, masks, and labels
- Robust error handling for missing or corrupted files

**Metrics Calculation**:
- Intersection over Union (IoU) for segmentation quality
- Pixel-wise precision, recall, and F1 score
- Detection rate and false positive/negative rates
- Confidence distribution analysis

**Visualization Generation**:
- IoU distribution histograms
- Precision-recall curves
- Confusion matrices
- Sample result visualizations with heatmaps

## Performance Metrics

**YOLOv8x Detection Baseline**:
- mAP50-95: 0.0211
- mAP50: 0.0693
- Precision: 0.0036
- Recall: 0.9516

**After Precision Improvement**:
- Significant reduction in false positives
- Improved precision while maintaining detection capabilities
- Enhanced segmentation accuracy with UNet integration 

## Implementation Details

```python
# 1. Initialize the detector
detector = CavityDetector(
    yolo_model_path='path/to/yolo_weights.pt',
    unet_model_path='path/to/unet_weights.pth',
    confidence_threshold=0.35,
    mask_threshold=0.45
)

# 2. Process a single image
results = detector.detect_and_segment('path/to/dental_xray.jpg', return_visualization=True)

# 3. Visualize results
detector.visualize_results(results, save_path='detection_result.png')

# 4. Evaluate on a dataset
evaluator = DatasetEvaluator(input_dir='dataset_path', output_dir='results_path')
metrics, detailed_results = evaluator.evaluate_model(detector, num_samples=100)
