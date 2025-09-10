# Research Summary: Hedgerow Detection Using Deep Learning

## Executive Summary

This research addresses the critical need for automated hedgerow mapping in environmental monitoring and land management. By applying state-of-the-art deep learning techniques to Very High-Resolution (VHR) satellite imagery, the study provides a comprehensive comparison of 34 different model architectures and establishes new benchmarks for hedgerow detection accuracy.

## Problem Statement

Traditional hedgerow mapping relies on labor-intensive ground surveys that are impractical for large-scale environmental monitoring. With hedgerows being vital for biodiversity conservation, soil erosion prevention, and habitat connectivity, there is an urgent need for automated, scalable mapping solutions.

## Research Objectives

1. Develop and train deep learning models for automated hedgerow detection
2. Compare performance across different architectures (CNNs vs Vision Transformers)
3. Evaluate transfer learning potential between aerial photography and satellite imagery
4. Establish benchmarks for future hedgerow detection research

## Methodology Overview

### Data Sources
- **BlueSky National Hedgerow Map**: 49 high-quality aerial images with ground truth masks
- **Airbus Pléiades Neo**: 1.2m resolution multispectral VHR satellite imagery
- **Processing**: 256×256 pixel tiles with comprehensive data augmentation

### Model Architectures Tested
- **U-Net variants** (ResNet, ResNeXt, ResNest backbones)
- **Feature Pyramid Networks (FPN)**
- **Pyramid Scene Parsing Networks (PSPNet)**
- **LinkNet**
- **Vision Transformers (ViT-MiT)**

### Training Strategy
- Combined loss function optimized for class imbalance and boundary precision
- AdamW optimizer with adaptive learning rate scheduling
- Early stopping and regularization to prevent overfitting
- Rigorous evaluation using IoU, precision, recall, and F1-score metrics

## Key Findings

### Model Performance Rankings
1. **U-Net ResNeXt50**: Best IoU (0.718) with balanced performance
2. **U-Net ResNest200e**: Highest recall (0.874) for comprehensive detection
3. **U-Net ResNet101**: Strong performance with extended training

### Architecture Insights
- **U-Net dominance**: Consistently outperformed other architectures across metrics
- **Skip connections crucial**: Encoder-decoder structure with skip connections proved essential for boundary delineation
- **Data efficiency**: U-Net architectures performed well even with limited training data
- **Vision Transformers**: Showed promise but require larger datasets for optimal performance

### Transfer Learning Challenges
- Spectral differences between aerial photography (RGB) and satellite imagery (multispectral)
- Pixel value range variations requiring careful normalization
- Resolution and perspective differences affecting feature recognition
- Domain adaptation requirements for operational deployment

## Implications and Applications

### Environmental Monitoring
- Large-scale hedgerow mapping for biodiversity assessments
- Monitoring hedgerow loss and fragmentation over time
- Supporting conservation planning and policy development

### Agricultural Management
- Precision agriculture applications
- Land use planning and compliance monitoring
- Supporting agri-environmental schemes

### Research Contributions
- First comprehensive comparison of deep learning architectures for hedgerow detection
- Establishment of performance benchmarks using standardized datasets
- Identification of transfer learning challenges and potential solutions

## Limitations and Future Directions

### Current Limitations
- Limited training dataset size (49 high-quality images)
- Domain gap between aerial photography and satellite imagery
- Computational requirements for processing large-scale imagery

### Future Research Opportunities
1. **Enhanced datasets**: Incorporation of improved UKCEH ground truth data
2. **Domain adaptation**: Advanced techniques for bridging aerial-satellite imagery gaps
3. **Multi-modal fusion**: Combining VHR imagery with LiDAR and other data sources
4. **Real-time processing**: Optimization for operational deployment scenarios

## Technical Innovations

### Loss Function Design
Novel combination of Dice, Focal, and BCE boundary losses specifically optimized for hedgerow segmentation tasks, addressing class imbalance and boundary precision requirements.

### Data Augmentation Strategy
Comprehensive augmentation pipeline including geometric transformations, photometric adjustments, and noise injection to improve model robustness.

### Evaluation Framework
Multi-metric evaluation approach prioritizing both detection accuracy (IoU) and completeness (recall) for practical applications.

## Broader Impact

This research provides the foundation for operational hedgerow monitoring systems that can support:
- **Policy makers**: Evidence-based conservation strategies
- **Land managers**: Efficient monitoring and compliance tools
- **Researchers**: Standardized benchmarks for future studies
- **Technology developers**: Proven architectures for similar environmental applications

The work demonstrates how advanced AI techniques can address real-world environmental challenges, contributing to sustainable land management and biodiversity conservation efforts.