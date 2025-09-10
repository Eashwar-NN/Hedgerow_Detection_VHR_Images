# Research Results and Visualizations

This directory contains key visualizations and results from the hedgerow detection research. The images demonstrate the comparative performance of 34 different deep learning architectures across multiple evaluation metrics.

## Directory Structure

```
visualizations/
├── training_curves/          # Validation loss curves across epochs
├── performance_metrics/      # IoU and recall comparisons  
├── predictions/             # Model prediction examples and comparisons
├── data_examples/           # Sample datasets and preprocessing results
```

## Key Findings Summary

### Top Performing Models
1. **U-Net ResNeXt50**: IoU = 0.718, Recall = 0.844
2. **U-Net ResNest200e**: IoU = 0.690, Recall = 0.874 (highest recall)
3. **U-Net ResNet101**: IoU = 0.703, Recall = 0.859 (longest trained)

### Architecture Performance Rankings
| Architecture | Best IoU | Best Recall | Training Efficiency |
|-------------|----------|-------------|-------------------|
| U-Net | 0.718 | 0.874 | High |
| LinkNet | 0.695 | 0.844 | Very High |
| FPN | 0.692 | 0.872 | Medium |
| ViT | 0.683 | 0.846 | Low |
| PSPNet | 0.630 | 0.835 | Medium |

## Visualization Descriptions

### Training Curves (`training_curves/`)

#### Validation Loss Curves
- **Purpose**: Monitor model convergence and identify overfitting
- **Key Insight**: U-Net models show most stable convergence patterns
- **Notable**: Early convergence in simpler architectures vs. extended training needs for complex models

#### Learning Rate Schedules
- **Configuration**: ReduceLROnPlateau scheduler with patience=25
- **Effect**: Adaptive learning rate reduction when validation loss plateaus
- **Impact**: Critical for fine-tuning model performance in later training epochs

### Performance Metrics (`performance_metrics/`)

#### IoU Curves by Architecture
- **Measurement**: Intersection over Union tracking across training epochs
- **Best Performance**: U-Net ResNeXt50 achieving 0.718 IoU
- **Convergence Pattern**: Most architectures plateau around epoch 50-100
- **Architecture Comparison**: Clear superiority of U-Net family architectures

#### Recall Curves Analysis
- **Critical Metric**: Recall prioritized for comprehensive hedgerow detection
- **Peak Performance**: U-Net ResNest200e with 0.874 recall
- **Trade-offs**: Higher recall models sometimes sacrifice precision
- **Application Relevance**: High recall crucial for ecological monitoring applications

#### Model Performance Summary
- **Comprehensive Comparison**: All 34 models evaluated on consistent metrics
- **Performance Distribution**: U-Net variants dominate top rankings
- **Architectural Insights**: Skip connections prove crucial for boundary delineation tasks

### Prediction Examples (`predictions/`)

#### Original vs Ground Truth vs Predictions
- **Visualization**: Side-by-side comparison of input imagery, ground truth masks, and model predictions
- **Top 3 Models**: Direct comparison of best performing architectures
- **Quality Assessment**: Visual validation of quantitative metrics
- **Boundary Accuracy**: Demonstration of precise hedgerow edge detection

#### Transfer Learning Results
- **Success Cases**: Examples where BlueSky-trained models perform well on Airbus imagery
- **Failure Cases**: Instances highlighting domain adaptation challenges
- **Domain Gap Analysis**: Spectral and resolution differences between aerial photography and satellite imagery
- **Lessons Learned**: Need for domain-specific fine-tuning approaches

### Data Examples (`data_examples/`)

#### Dataset Samples
- **BlueSky Data**: High-resolution aerial photography with ground truth annotations
- **Airbus Pléiades Neo**: Multispectral VHR satellite imagery samples
- **Quality Control**: Examples of data filtering and quality assessment process
- **Preprocessing Pipeline**: Tiling, normalization, and augmentation examples

#### Data Augmentation Demonstrations
- **Geometric Transformations**: Rotation, flipping, and scaling examples
- **Photometric Adjustments**: Brightness, contrast, and color variations
- **Noise Addition**: Gaussian noise and dropout examples
- **Impact Assessment**: Before/after augmentation model performance comparisons

## Experimental Validation

### Statistical Significance
- **Multiple Runs**: Consistent results across training iterations
- **Cross-Validation**: Robust performance on held-out validation data
- **Metric Reliability**: Strong correlation between different evaluation measures

### Model Selection Criteria
1. **Primary**: Intersection over Union (IoU) for segmentation accuracy
2. **Secondary**: Recall for comprehensive hedgerow detection
3. **Tertiary**: Training efficiency and computational requirements

## Key Research Insights

### Why U-Net Excelled
- **Skip Connections**: Preserve spatial information during upsampling
- **Data Efficiency**: Effective performance with limited training data (49 high-quality images)
- **Boundary Preservation**: Superior edge detection compared to other architectures
- **Architectural Balance**: Optimal trade-off between local detail and global context

### Vision Transformer Limitations
- **Data Requirements**: Need larger datasets for optimal performance
- **Spatial Relationships**: Less effective at capturing local spatial dependencies
- **Training Complexity**: Longer convergence times and higher computational costs
- **Domain Specificity**: Better suited for natural image tasks than specialized segmentation

### Transfer Learning Challenges
- **Spectral Mismatch**: 3-channel (RGB) vs 4-channel (multispectral) imagery
- **Scale Differences**: Aerial photography vs satellite imagery resolution variations
- **Perspective Variations**: Different viewing angles and image acquisition conditions
- **Domain Adaptation Needs**: Requirement for specialized techniques to bridge data source gaps

## Implications for Operational Deployment

### Recommended Architecture
- **Production Use**: U-Net ResNeXt50 for balanced performance
- **High Recall Applications**: U-Net ResNest200e for comprehensive detection
- **Resource-Constrained Environments**: LinkNet ResNet34 for computational efficiency

### Scalability Considerations
- **Large-Scale Mapping**: Batch processing capabilities validated
- **Real-Time Applications**: LinkNet architectures suitable for time-sensitive applications
- **Cloud Deployment**: All models compatible with standard ML deployment frameworks

## Future Research Directions

### Model Improvements
- **Ensemble Methods**: Combining top-performing architectures
- **Attention Mechanisms**: Integration of spatial attention modules
- **Multi-Scale Training**: Enhanced handling of varying hedgerow sizes

### Data Enhancement
- **Domain Adaptation**: Advanced techniques for cross-domain transfer
- **Multi-Modal Fusion**: Integration of LiDAR and optical imagery
- **Temporal Analysis**: Time-series hedgerow change detection

### Operational Integration
- **API Development**: RESTful services for model deployment
- **Real-Time Processing**: Streaming data processing capabilities
- **Quality Assurance**: Automated prediction confidence assessment

---

**Note**: Detailed quantitative results and statistical analyses are available in the complete dissertation document. These visualizations provide a comprehensive overview of the experimental methodology and key findings from this research.