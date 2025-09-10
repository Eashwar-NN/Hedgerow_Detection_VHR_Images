# Hedgerow Detection in Very High-Resolution Satellite Imagery using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

> **MSc Artificial Intelligence Dissertation - University of Stirling (2024)**  
> **Grade: Distinction**

This repository contains the research outcomes and analysis from my dissertation on automated hedgerow detection in Very High-Resolution (VHR) satellite imagery using deep learning techniques. The complete implementation will be made available following academic publication.

## 📄 Research Overview

Hedgerows are vital linear landscape features that support biodiversity, prevent soil erosion, and facilitate habitat connectivity. Traditional ground surveys for mapping these features are labor-intensive and impractical for large-scale assessments. This research demonstrates the efficacy of deep learning techniques, specifically CNNs (U-Net, FPN, PSPNet, LinkNet) and Vision Transformers, for automated hedgerow detection in VHR satellite imagery.

**Key Achievements:**
- Trained and evaluated 34 distinct model architectures
- Achieved best IoU of 0.718 with U-Net ResNeXt50 architecture  
- Successfully demonstrated transfer learning between aerial and satellite imagery
- Provided comprehensive comparative analysis of different deep learning approaches

## 📊 Key Results

| Model | Architecture | IoU | Recall | F1-Score | Precision |
|-------|--------------|-----|--------|----------|-----------|
| **U-Net ResNeXt50** | U-Net | **0.718** | 0.844 | 0.836 | 0.828 |
| **U-Net ResNest200e** | U-Net | 0.690 | **0.874** | 0.817 | 0.767 |
| **U-Net ResNet101** | U-Net | 0.703 | 0.859 | 0.825 | 0.794 |

## 🔬 Research Methodology

### Dataset Information
- **Primary Dataset**: BlueSky National Hedgerow Map (aerial photography with ground truth masks)
- **Secondary Dataset**: Airbus Pléiades Neo (1.2m resolution multispectral VHR satellite imagery)
- **Processing**: Images tiled to 256x256 pixels with comprehensive data augmentation

### Model Architectures Evaluated
1. **Convolutional Neural Networks (CNNs)**:
   - U-Net (multiple backbone variants)
   - Feature Pyramid Networks (FPN)
   - Pyramid Scene Parsing Networks (PSPNet)
   - LinkNet
   
2. **Vision Transformers (ViTs)**:
   - Mix Transformer (MiT) configurations

### Training Configuration
- Combined loss function: Dice (0.3) + Focal (0.4) + BCE Boundary (0.3)
- AdamW optimizer with ReduceLROnPlateau scheduler
- Early stopping and dropout regularization
- 500 maximum epochs with comprehensive evaluation metrics

## 📈 Research Contributions

1. **Comprehensive Model Comparison**: Systematic evaluation of 34 different architectures for hedgerow detection
2. **Transfer Learning Analysis**: First study applying hedgerow detection across different imagery types (aerial to satellite)
3. **Optimized Training Pipeline**: Combined loss function specifically designed for hedgerow segmentation
4. **Practical Validation**: Real-world testing on both aerial photography and VHR satellite imagery

## 🎯 Key Findings

- **U-Net architectures consistently outperformed** other approaches, likely due to their encoder-decoder structure with skip connections being well-suited for boundary delineation tasks
- **Data efficiency was crucial** - U-Net's ability to work well with limited training data (49 high-quality images after augmentation) was a significant advantage
- **Transfer learning challenges** identified between aerial photography and satellite imagery domains, highlighting the need for domain adaptation techniques

## 📚 Academic Recognition

- **Degree**: Master of Science in Artificial Intelligence
- **Institution**: University of Stirling, Scotland
- **Grade**: Distinction
- **Supervisor**: Dr. Vahid Akbari
- **Collaboration**: University College Dublin (UCD)

## 🔄 Publication Status

This research is currently being prepared for academic publication. The complete implementation code and detailed methodology will be made available in this repository following publication to ensure compliance with academic publishing requirements.

**Watch this repository** for updates on code availability and publication status.

## 📊 Available Resources

- [Complete Dissertation PDF](docs/dissertation.pdf) - Full research document with detailed methodology and results
- [Research Summary](docs/research_summary.md) - Concise overview of findings and implications
- [Model Performance Analysis](docs/model_comparison.md) - Detailed comparison of all 34 architectures tested
- [Experiment Tracking](https://api.wandb.ai/links/eashwar408-university-of-stirling/amhlp70h) - Complete training logs and metrics

## 🤝 Acknowledgments

- **Dr. Vahid Akbari** - Research Supervisor, University of Stirling
- **Mehran Alizadeh Pirbasti** - Collaborator, University College Dublin
- **UK Centre for Ecology & Hydrology (UKCEH)** - Hedgerow map data provision
- **Airbus** - VHR satellite imagery access
- **BlueSky International** - Aerial hedgerow dataset and ground truth masks

## 📖 Citation

If you reference this work in your research, please cite:

```bibtex
@mastersthesis{nithyameenakshi2024hedgerow,
  title={Developing Deep Learning Techniques for Hedgerow Detection in Very High-Resolution Satellite Imagery},
  author={Nithyameenakshi, Eashwar Nagarajan},
  year={2024},
  school={University of Stirling},
  type={Master's thesis}
}
```

## 📧 Contact

For research inquiries or collaboration opportunities:

**Eashwar Nagarajan Nithyameenakshi**  
- Email: eashwar408@gmail.com
- LinkedIn: https://www.linkedin.com/in/eashwarnn

## 🔗 Related Links

- [University of Stirling - Computing Science](https://www.stir.ac.uk/about/faculties/natural-sciences/computing-science-and-mathematics/)
- [BlueSky National Hedgerow Map](https://bluesky-world.com/national-hedgerow-map/)

---

**Implementation code will be made available following academic publication. Watch this space for updates!**