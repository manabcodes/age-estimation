# Age Estimation for Child Protection: A Comprehensive Study of Deep Learning Approaches

**A Master's Thesis Project for Panacea Cooperative**  
*European Master in Law, Data, and AI (EMILDAI) - Universidad de León*  
*November 2024 - June 2025*

## Overview

This repository contains the complete implementation and analysis for a comprehensive age estimation research project conducted in collaboration with [Panacea Cooperative](https://panacea-coop.com/). The research systematically evaluates thirteen distinct age estimation models across multiple architectural paradigms and loss function formulations, with a particular focus on applications in child protection contexts.

### Key Contributions

- **Comprehensive Model Evaluation**: Systematic comparison of 13 age estimation models across 4 architectural paradigms
- **Quality-Aware Dataset Processing**: Novel unified dataset creation pipeline using OFIQ quality metrics
- **Bias Mitigation**: Identification and mitigation of age-quality and gender-quality correlations in existing datasets
- **Practical Applications**: Focus on child protection and digital safety applications

## Project Structure

```
├── README.md
├── filter_dataset/              # Dataset processing and filtering
│   ├── filter_datasets.py      # Quality-based filtering implementation
│   └── train_test_split.py     # Unified dataset creation pipeline
├── ofiq_visualizations/         # Quality analysis and visualization
│   ├── visualize_distributions.py       # OFIQ metric analysis
│   ├── visualize_gender_age_correlations.py  # Bias analysis
│   ├── visualize_pose.py        # Head pose analysis
│   └── visualize_tiles_metrics.py       # Quality visualization
├── resnet_codes/               # Model implementations
│   ├── resnet_cross_entropy.py # Cross-entropy baseline
│   ├── resnet_dldlv2.py        # DLDL-v2 implementation
│   └── [other model variants]  # Additional architectures
└── docs/                       # Documentation and reports
    ├── OFIQ_Report.pdf         # Comprehensive quality analysis
    └── thesis_document.pdf     # Complete thesis (when available)
```

## Dataset Processing Pipeline

Our research introduces a systematic three-step dataset processing methodology:

### Step 1: Quality-Based Filtering
- **OFIQ Quality Assessment**: Comprehensive analysis using ISO/IEC 29794-5 standard
- **Adaptive Thresholds**: Dataset-specific quality requirements
- **Aggressive Filtering**: 65% reduction removing poor sharpness, expression, and illumination

### Step 2: Balanced Sampling
- **Pose Variation Balance**: Systematic sampling across head pose angles
- **Quality Condition Balance**: Ensuring diverse quality conditions
- **Additional 42% Reduction**: Focus on representative samples

### Step 3: Bias Mitigation
- **Age-Quality Correlation**: Addressing systematic quality variations across age groups
- **Gender-Quality Correlation**: Mitigating gender-based quality biases
- **Final Dataset**: 3,427 high-quality, demographically balanced images

## Technical Implementation

### Model Architectures
- **ResNet-50**: Standard CNN baseline with progressive unfreezing
- **Vision Transformer (ViT)**: Attention-based architecture
- **CNN-Transformer Hybrid**: Combined convolutional and attention mechanisms
- **Prompt-based Models**: Novel prompt engineering approaches

### Loss Functions Evaluated
- **Cross-Entropy**: Standard classification baseline
- **SORD (Soft Ordinal)**: Ordinal-aware soft label distribution
- **DLDL-v2**: Deep Label Distribution Learning
- **Mean-Variance**: Probabilistic age estimation
- **Focal Loss**: Addressing class imbalance

### Training Strategy
- **Progressive Unfreezing**: Gradual layer unfreezing for transfer learning
- **Quality-Aware Sampling**: Training data selection based on OFIQ metrics
- **Comprehensive Evaluation**: Multiple metrics including MAE, accuracy, and calibration

## Key Research Findings

### Dataset Quality Analysis
- **Systematic Quality Issues**: 99.54% of original images required filtering
- **Dataset-Specific Patterns**: Quality varies significantly across datasets
- **Bias Discovery**: Strong correlations between age/gender and image quality

### Model Performance Insights
- **Architecture Impact**: Transformers and Resnet shows minimal difference
- **Loss Function Effectiveness**: SORD and DLDL-v2 outperform standard cross-entropy
- **Quality Sensitivity**: Model performance heavily dependent on input image quality
- **FaRL Weights**: Superior visual-language information useful as pre-trained weights
- **PoE Reliability**: PoE was the only method that gave reliable uncertainty estimates

### Practical Implications
- **Child Protection Applications**: Reliable age verification for digital platforms
- **Quality Requirements**: Minimum quality thresholds for deployment
- **Fairness Considerations**: Bias mitigation essential for ethical deployment

## Installation and Usage

### Prerequisites
```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
tqdm>=4.62.0

# OFIQ quality assessment (if needed)
# Follow from here: https://github.com/BSI-OFIQ/OFIQ-Project/tree/main
```

### Quick Start
```bash
# Clone the repository
git clone [repository-url]
cd age-estimation-thesis

# Install dependencies
pip install -r requirements.txt

# Run dataset processing pipeline
python filter_dataset/train_test_split.py

# Train baseline ResNet model
python resnet_codes/resnet_cross_entropy.py

# Generate quality analysis visualizations
python ofiq_visualizations/visualize_distributions.py
```

### Configuration
Update the configuration paths in the training scripts:
```python
config = {
    'train_csv': '/path/to/train_annotations.csv',
    'val_csv': '/path/to/val_annotations.csv',
    'train_dir': '/path/to/train',
    'val_dir': '/path/to/val',
    # ... other parameters
}
```

## Evaluation Metrics

Our comprehensive evaluation framework includes:

- **Mean Absolute Error (MAE)**: Primary age estimation accuracy metric
- **Accuracy@±N**: Percentage of predictions within N years
- **Demographic Fairness**: Performance across age groups and genders
- **Uncertainty Estimate**: For calibrating performances


## Ethical Considerations

This research addresses several critical ethical dimensions:

- **Child Protection**: Robust age verification for digital safety
- **Privacy Preservation**: Minimal data retention and processing
- **Fairness and Bias**: Systematic bias identification and mitigation
- **Deployment Guidelines**: Responsible AI deployment recommendations

## Research Timeline

- **November-December 2024**: Literature review and methodology development
- **January-February 2025**: Dataset processing and quality analysis
- **March-April 2025**: Model implementation and training
- **April-May 2025**: Comprehensive evaluation and analysis
- **June 2025**: Documentation and thesis completion

## Collaboration

This research was conducted through a collaborative framework between:

- **Universidad de León**: Academic oversight and research infrastructure
- **Panacea Cooperative**: Industry partnership and practical guidance
- **EMILDAI Program**: International academic context and standards

## License and Usage

This research is conducted for academic purposes as part of a Master's thesis program. The code and methodologies are available for research and educational use. Please cite appropriately if using components of this work.

## Contact

For questions about this research or collaboration opportunities:

- **Author**: MEEM ARAFAT MANAB
- **Institution**: Universidad de León - EMILDAI Program  
- **Partner**: Panacea Cooperative
- **Email**: meem.arafat@bracu.ac.bd

## Acknowledgments

We thank Panacea Cooperative for their collaboration and practical insights, Universidad de León for academic support and infrastructure, the EMILDAI program for the international research context, Dr. Victor Gonzalez Castro, Dr. Guillermo Gomez Trenado, and Dr. Oscar Ibañez for their continuous supervision, and the open-source community for tools and frameworks that enabled this research.

---

*This repository represents six months of comprehensive research into age estimation methodologies, dataset quality analysis, and practical deployment considerations for child protection applications.*
