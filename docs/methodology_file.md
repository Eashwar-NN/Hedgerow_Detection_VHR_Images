# Methodology: Deep Learning for Hedgerow Detection

## Research Design

This study employs a comparative experimental design to evaluate the effectiveness of different deep learning architectures for automated hedgerow detection in Very High-Resolution (VHR) satellite imagery. The methodology follows a systematic approach encompassing data collection, preprocessing, model development, training, and rigorous evaluation.

## Data Collection and Sources

### Primary Dataset: BlueSky National Hedgerow Map
- **Data Type**: True color aerial photography with corresponding hedgerow ground truth masks
- **Spatial Accuracy**: ±1m RMSE
- **Coverage**: National hedgerow mapping dataset for the UK
- **Quality Control**: Manual inspection and filtering resulting in 49 high-quality image-mask pairs
- **Ground Truth**: Binary masks indicating hedgerow presence/absence at pixel level

### Secondary Dataset: Airbus Pléiades Neo
- **Data Type**: Multispectral VHR satellite imagery
- **Spatial Resolution**: 1.2 meters
- **Spectral Bands**: Red, Green, Blue, Near-infrared (4 channels)
- **Usage**: Transfer learning validation and domain adaptation testing
- **Processing**: Converted to 3-channel RGB for compatibility with BlueSky-trained models

### Validation Dataset: UKCEH Hedgerow Map
- **Data Type**: LiDAR-derived hedgerow mapping
- **Coverage**: England-wide hedgerow detection
- **Accuracy**: Approximately 76% (acknowledged by UKCEH as requiring revision)
- **Usage**: Initial validation attempts (limited due to accuracy issues)

## Data Preprocessing Pipeline

### Image Tiling and Standardization
1. **Raster Processing**: Large-format imagery processed using QGIS
2. **Tile Generation**: Images divided into 256×256 pixel tiles
3. **Mask Alignment**: Corresponding ground truth masks processed identically
4. **Quality Assessment**: Manual inspection to remove tiles with mislabeled features

### Data Quality Control Measures
- Visual inspection of each image-mask pair
- Removal of tiles containing incorrect hedgerow annotations
- Filtering of tiles with insufficient hedgerow content
- Validation of mask-image spatial alignment

### Normalization and Standardization
- Pixel values normalized to [0,1] range
- Consistent color space conversion (RGB)
- Spatial coordinate system alignment across datasets

## Data Augmentation Strategy

### Geometric Transformations
- **RandomResizedCrop**: Scale and position variability
- **HorizontalFlip & VerticalFlip**: Orientation independence
- **RandomRotate90**: 90-degree rotational augmentation
- **ShiftScaleRotate**: Combined spatial transformations
- **ElasticTransform**: Local deformation simulation
- **GridDistortion**: Perspective distortion modeling

### Photometric Augmentations
- **RandomBrightnessContrast**: Illumination variation simulation
- **HueSaturationValue**: Color space variations
- **RandomGamma**: Gamma correction variations
- **GaussNoise**: Sensor noise simulation
- **ChannelShuffle**: Color channel robustness

### Advanced Augmentations
- **CoarseDropout**: Occlusion simulation
- Various dropout patterns to improve generalization

## Model Architecture Selection

### Convolutional Neural Networks (CNNs)

#### U-Net Architecture
- **Base Design**: Encoder-decoder with skip connections
- **Backbones Tested**: ResNet50, ResNet101, ResNeXt50, ResNeXt101, ResNest200e
- **Skip Connections**: Concatenation-based feature fusion
- **Output**: Single-channel probability maps

#### Feature Pyramid Networks (FPN)
- **Multi-scale Processing**: Hierarchical feature extraction
- **Lateral Connections**: Cross-scale feature fusion
- **Backbones**: ResNet and ResNeXt variants
- **Scale Handling**: Multiple resolution feature maps

#### Pyramid Scene Parsing Networks (PSPNet)
- **Global Context**: Pyramid pooling module
- **Scene Understanding**: Multi-scale context aggregation
- **Backbones**: ResNet and ResNeXt architectures
- **Pooling Strategy**: Multiple kernel sizes for context capture

#### LinkNet
- **Efficiency Focus**: Lightweight encoder-decoder design
- **Skip Connections**: Additive feature fusion
- **Backbones**: ResNet34, MobileNetV3
- **Computational Efficiency**: Optimized for inference speed

### Vision Transformers (ViTs)

#### Mix Transformer (MiT)
- **Architecture**: SegFormer-based implementation
- **Variants Tested**: MiT-B2, MiT-B4 with different decoders
- **Self-Attention**: Global context modeling
- **Patch Processing**: 16×16 pixel patches for transformer input

## Loss Function Design

### Combined Loss Function
The training employed a weighted combination of three complementary loss functions:

#### Dice Loss (Weight: 0.3)
- **Purpose**: Maximize overlap between prediction and ground truth
- **Advantage**: Handles class imbalance effectively
- **Formula**: 2×|X∩Y| / (|X|+|Y|)

#### Focal Loss (Weight: 0.4)
- **Purpose**: Address class imbalance by focusing on hard examples
- **Advantage**: Down-weights easy background pixels
- **Parameters**: α=0.25, γ=2.0

#### Binary Cross-Entropy Boundary Loss (Weight: 0.3)
- **Purpose**: Enhance boundary precision
- **Advantage**: Sharp edge delineation
- **Focus**: Pixel-level classification accuracy

### Loss Function Rationale
The weighted combination was optimized through empirical testing, with higher weight on focal loss to counteract observed class imbalance in the hedgerow dataset.

## Training Configuration

### Hyperparameters
- **Learning Rate**: 0.0001 (conservative approach for stable convergence)
- **Batch Size**: 8 (memory constraint optimization)
- **Maximum Epochs**: 500 (with early stopping)
- **Optimizer**: AdamW with weight decay (0.0001)
- **Scheduler**: ReduceLROnPlateau (patience=25, factor=0.5)

### Regularization Techniques
- **Dropout**: 0.33 in decoder layers
- **Weight Decay**: L2 regularization (0.0001)
- **Early Stopping**: IoU and accuracy-based (patience=30)

### Training Process
1. **Data Split**: 70% training, 30% validation (35 train, 14 validation images)
2. **Augmentation**: Applied only to training set
3. **Validation**: Performed after each epoch
4. **Checkpointing**: Best model saved based on validation IoU
5. **Monitoring**: Comprehensive metric tracking via Weights & Biases

## Evaluation Methodology

### Primary Metrics

#### Intersection over Union (IoU)
- **Definition**: |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
- **Significance**: Direct measure of segmentation accuracy
- **Usage**: Primary model selection criterion

#### Recall (Sensitivity)
- **Definition**: True Positives / (True Positives + False Negatives)
- **Significance**: Measures hedgerow detection completeness
- **Usage**: Critical for ecological applications

#### Precision
- **Definition**: True Positives / (True Positives + False Positives)
- **Significance**: Measures prediction accuracy
- **Usage**: Important for mapping quality

#### F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Significance**: Balanced performance measure
- **Usage**: Overall model performance assessment

### Secondary Metrics
- **Accuracy**: Overall pixel classification correctness
- **Training/Validation Loss**: Model convergence monitoring

### Model Selection Criteria
1. **Primary**: Highest IoU score
2. **Secondary**: Highest recall for comprehensive detection
3. **Tertiary**: Training duration vs. performance trade-off

## Transfer Learning Methodology

### Domain Adaptation Strategy
1. **Source Domain**: BlueSky aerial photography (3-channel RGB)
2. **Target Domain**: Airbus VHR satellite imagery (4-channel multispectral)
3. **Preprocessing**: Channel reduction (RGB extraction from multispectral)
4. **Normalization**: Pixel value range alignment (0-3858 → 0-255)

### Transfer Learning Process
1. **Pre-training**: Complete training on BlueSky dataset
2. **Model Loading**: Best checkpoint selection
3. **Fine-tuning**: Application to Airbus imagery
4. **Evaluation**: Qualitative assessment of prediction quality

### Challenges Addressed
- Spectral band differences between datasets
- Pixel value range variations
- Resolution and perspective differences
- Domain gap between aerial and satellite imagery

## Experimental Design

### Comparative Analysis Framework
- **Total Models**: 34 distinct architecture-backbone combinations
- **Controlled Variables**: Hyperparameters, training data, evaluation metrics
- **Variable Factors**: Architecture type, backbone network, model complexity

### Statistical Validation
- **Reproducibility**: Fixed random seeds for consistent results
- **Cross-validation**: Validation set performance for model selection
- **Multiple Runs**: Consistent performance verification

## Quality Assurance

### Data Quality Measures
- Manual inspection of all training data
- Ground truth validation against expert knowledge
- Systematic removal of mislabeled examples

### Model Validation
- Comprehensive metric tracking
- Visual inspection of predictions
- Cross-dataset validation (transfer learning)

### Reproducibility Standards
- Detailed hyperparameter documentation
- Code version control and documentation
- Experiment tracking via Weights & Biases platform

## Computational Resources

### Hardware Requirements
- GPU acceleration for deep learning training
- Sufficient memory for batch processing
- Storage for large-scale imagery datasets

### Software Environment
- PyTorch deep learning framework
- Segmentation Models PyTorch library
- QGIS for geospatial data processing
- Python ecosystem for scientific computing

This methodology ensures rigorous, reproducible research while addressing the specific challenges of hedgerow detection in VHR imagery through comprehensive model comparison and evaluation.