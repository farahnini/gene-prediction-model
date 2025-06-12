# Genome Annotation Model

A deep learning model for genome annotation using DNA-BERT architecture, trained on diverse species to predict functional elements in DNA sequences (Non-coding, Coding, Regulatory regions).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/genome-annotation-model.git
cd genome-annotation-model

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download genome annotations
python download_annotations.py

# Train the model (all organisms)
python model.py --epochs 10 --batch_size 32

# Train on single organism (faster for testing)
python model.py --organism "escherichia_coli" --epochs 5 --batch_size 16

# Evaluate the model
python evaluate.py --checkpoint checkpoints/best_model.pt

# Generate HTML evaluation report
python evaluate_display.py --results_file evaluation_results/comprehensive_evaluation.json
```

## Features

### 🧬 Advanced Model Architecture
- **DNA-BERT**: Pre-trained transformer architecture adapted for DNA sequences
- **Hybrid CNN-LSTM**: Convolutional layers for local patterns + LSTM for sequence context
- **Attention Mechanism**: Focus on relevant genomic regions
- **Multi-class Classification**: Predicts Non-coding, Coding, and Regulatory elements

### 📊 Comprehensive Training Features
- **Real-time Progress Tracking**: Live progress bars with ETA, batch speed, and metrics
- **System Monitoring**: CPU, memory, and GPU usage tracking
- **Per-epoch Checkpointing**: Automatic model saving after each epoch
- **TensorBoard Integration**: Real-time training visualization
- **Single Organism Training**: Train on specific species for faster experimentation

### 🔍 Advanced Evaluation & Visualization
- **Detailed Metrics**: Accuracy, precision, recall, F1-score per class and overall
- **Interactive HTML Reports**: Beautiful, responsive evaluation reports
- **Training History Visualization**: Loss curves, accuracy plots, learning rate schedules
- **Confusion Matrix Analysis**: Visual representation of model predictions
- **Per-class Performance**: Detailed breakdown by annotation type

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git
- CUDA-capable GPU (recommended for faster training)
- 16GB RAM minimum
- 50GB free disk space

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
# Clone the repository to your local machine
git clone https://github.com/yourusername/genome-annotation-model.git

# Navigate to the project directory
cd genome-annotation-model
```

#### 2. Set Up Python Environment

Choose one of the following methods:

##### Option A: Using venv (Recommended)
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.8 or higher
```

##### Option B: Using conda
```bash
# Create a new conda environment
conda create -n genome-annotation python=3.8

# Activate the environment
conda activate genome-annotation

# Verify Python version
python --version  # Should show Python 3.8 or higher
```

#### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

#### 4. Verify GPU Support (Optional but Recommended)
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA is available, check GPU information
python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"}')"
```

#### 5. Download Genome Annotations
```bash
# Edit the email in download_annotations.py
# Replace "your.email@example.com" with your email address

# Run the download script
python download_annotations.py
```

### Common Installation Issues and Solutions

#### 1. CUDA Installation Issues
- **Problem**: CUDA not found
  - **Solution**: Install NVIDIA CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
  - **Verify**: Run `nvidia-smi` in terminal

#### 2. Python Version Issues
- **Problem**: Python version too old
  - **Solution**: Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
  - **Verify**: Run `python --version`

#### 3. Package Installation Issues
- **Problem**: pip install fails
  - **Solution**: Update pip first: `python -m pip install --upgrade pip`
  - **Alternative**: Try installing packages one by one

#### 4. Virtual Environment Issues
- **Problem**: venv not found
  - **Solution**: Install venv: `python -m pip install virtualenv`
  - **Alternative**: Use conda instead

#### 5. Memory Issues
- **Problem**: Out of memory during installation
  - **Solution**: Close other applications
  - **Alternative**: Install packages with `--no-cache-dir` flag

### Post-Installation Verification

Run these commands to verify your installation:

```bash
# Check Python environment
python --version
pip list

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Test the model
python model.py --test
```

### Directory Structure After Installation
```
genome-annotation-model/
├── venv/                  # Virtual environment
├── data/                  # Downloaded genome annotations
├── models/               # Saved model checkpoints
├── results/              # Evaluation results
│   └── reports/         # HTML evaluation reports
├── src/                  # Source code
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Dataset Overview

The model is trained on genome annotations from 30 diverse species across multiple lineages:

| Lineage | Species | Common Name |
|---------|---------|-------------|
| **Mammals** | Homo sapiens | Human |
| | Mus musculus | Mouse |
| | Rattus norvegicus | Rat |
| | Pan troglodytes | Chimpanzee |
| **Birds** | Gallus gallus | Chicken |
| | Taeniopygia guttata | Zebra finch |
| | Meleagris gallopavo | Turkey |
| **Fish** | Danio rerio | Zebrafish |
| | Oryzias latipes | Japanese medaka |
| | Gasterosteus aculeatus | Stickleback |
| **Insects** | Drosophila melanogaster | Fruit fly |
| | Apis mellifera | Honey bee |
| | Bombyx mori | Silkworm |
| **Plants** | Arabidopsis thaliana | Thale cress |
| | Oryza sativa | Rice |
| | Zea mays | Maize |
| | Glycine max | Soybean |
| **Fungi** | Saccharomyces cerevisiae | Baker's yeast |
| | Schizosaccharomyces pombe | Fission yeast |
| | Aspergillus nidulans | Aspergillus |
| **Bacteria** | Escherichia coli | E. coli |
| | Bacillus subtilis | B. subtilis |
| | Pseudomonas aeruginosa | P. aeruginosa |
| | Mycobacterium tuberculosis | M. tuberculosis |
| **Archaea** | Methanocaldococcus jannaschii | Methanococcus |
| | Sulfolobus solfataricus | Sulfolobus |
| | Halobacterium salinarum | Halobacterium |
| **Nematodes** | Caenorhabditis elegans | C. elegans |
| | Caenorhabditis briggsae | C. briggsae |
| **Amphibians** | Xenopus tropicalis | Western clawed frog |
| | Xenopus laevis | African clawed frog |
| **Reptiles** | Anolis carolinensis | Green anole |
| | Python bivittatus | Burmese python |

## Model Architecture

The **DNABERTModel** uses a sophisticated hybrid architecture:

```
Input DNA Sequence (1000 bp)
    ↓
DNA-BERT Encoder (12 layers, 768 hidden size)
    ↓ 
Convolutional Layers (Pattern Recognition)
    ├── Conv1D (768 → 256, kernel=3)
    ├── BatchNorm + ReLU
    ├── Conv1D (256 → 128, kernel=3)
    └── BatchNorm + ReLU
    ↓
Bidirectional LSTM (Sequence Context)
    ├── 2 layers, 64 hidden units each
    └── Dropout (0.1)
    ↓
Attention Mechanism (Focus on Important Regions)
    ├── Linear (128 → 64)
    ├── Tanh activation
    └── Linear (64 → 1) + Softmax
    ↓
Classification Head
    ├── Linear (128 → 64)
    ├── ReLU + Dropout
    └── Linear (64 → 3) [Non-coding, Coding, Regulatory]
```

### Key Components:
- **DNA-BERT**: Transformer encoder pre-trained on DNA sequences
- **CNN Layers**: Local pattern recognition (motifs, regulatory elements)
- **LSTM Layers**: Long-range dependencies and sequence context
- **Attention**: Weighted importance of different sequence regions
- **Classification**: Multi-class prediction with probability outputs

## Model Comparison

We evaluate multiple model architectures to find the optimal solution:

1. **CNN-LSTM Hybrid** (Default)
   - Convolutional layers for local patterns
   - Bidirectional LSTM for sequence context
   - Attention mechanism
   - Best for general-purpose annotation

2. **Transformer-based**
   - Self-attention mechanism
   - Position-wise feed-forward networks
   - Best for long-range dependencies

3. **ResNet-based**
   - Deep residual networks
   - Skip connections
   - Best for complex pattern recognition

4. **EfficientNet-based**
   - Compound scaling
   - Mobile-friendly architecture
   - Best for resource-constrained environments

To compare models:
```bash
python compare_models.py
```

This will:
1. Train each model architecture
2. Evaluate performance metrics
3. Generate comparison report
4. Save the best model

## Evaluation

The model evaluation includes:
- Accuracy, precision, recall, and F1 score
- Per-species performance metrics
- Confusion matrix analysis
- Error rate analysis

Results are presented in an interactive HTML report with:
- Interactive visualizations
- Detailed performance metrics
- Species-wise comparisons
- Error analysis

## Usage

### 1. Download Genome Annotations
```bash
# Edit your email in download_annotations.py first
python download_annotations.py
```

### 2. Training Options

#### Train on All Organisms (Comprehensive)
```bash
# Full training with default parameters
python model.py --epochs 20 --batch_size 32 --learning_rate 1e-4

# With custom settings
python model.py \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --sequence_length 1000 \
    --save_dir my_checkpoints
```

#### Train on Single Organism (Fast Testing)
```bash
# Quick training on E. coli
python model.py --organism "escherichia_coli" --epochs 10

# Train on human data
python model.py --organism "homo_sapiens" --epochs 15 --batch_size 16

# Train on plant data
python model.py --organism "arabidopsis_thaliana" --epochs 12
```

#### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--learning_rate` | 1e-4 | Learning rate for optimizer |
| `--sequence_length` | 1000 | DNA sequence length (bp) |
| `--organism` | None | Single organism name (optional) |
| `--save_dir` | checkpoints | Directory for saving models |
| `--data_dir` | genome_annotations | Data directory path |

### 3. Real-time Training Monitoring

During training, you'll see:
```
============================================================
SYSTEM INFORMATION
============================================================
Platform: Windows-10-10.0.19044-SP0
Processor: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD
CPU Cores: 16
Total Memory: 32.00 GB
GPU Available: True
GPU Name: NVIDIA GeForce RTX 3080
GPU Memory: 10.00 GB
============================================================

Processing species: 100%|██████████| 5/5 [02:30<00:00, 30.1s/species]

================================================================================
EPOCH 1/10
================================================================================
System Resources:
  CPU Usage: 45.2%
  Memory Usage: 8.45 GB
  GPU Memory: 3.21 GB allocated, 4.15 GB reserved

Training Epoch  1: 100%|██████████| 156/156 [03:45<00:00, 1.4batch/s, Loss=0.8234, Acc=0.6789, LR=1.00e-04, ETA=00:00:00]
Validation Epoch  1: 100%|██████████| 39/39 [00:32<00:00, 2.1batch/s, Loss=0.7123, Acc=0.7234, ETA=00:00:00]

EPOCH 1 RESULTS:
  Time: 258.34s (4.3 min)
  Train - Loss: 0.823456, Accuracy: 0.678934
  Valid - Loss: 0.712345, Accuracy: 0.723456
  Learning Rate: 1.00e-04
  ETA for completion: 02:15:30
  ★ NEW BEST MODEL! Validation accuracy: 0.723456
  Checkpoint saved: checkpoints/checkpoint_epoch_001.pt
```

### 4. Evaluation

#### Basic Evaluation
```bash
# Evaluate best model
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate specific checkpoint
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_020.pt

# Evaluate on single organism
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --organism "escherichia_coli" \
    --output_dir my_results
```

#### Evaluation Output
```
============================================================
EVALUATION RESULTS
============================================================
Overall Accuracy: 0.8567
Overall Precision: 0.8234
Overall Recall: 0.8756
Overall F1-Score: 0.8489

Per-Class Results:
Non-coding:
  Precision: 0.8123
  Recall: 0.8456
  F1-Score: 0.8287
  Support: 1234
Coding:
  Precision: 0.8567
  Recall: 0.8234
  F1-Score: 0.8398
  Support: 2345
Regulatory:
  Precision: 0.7989
  Recall: 0.8567
  F1-Score: 0.8267
  Support: 876
```

### 5. Generate HTML Reports
```bash
# Generate interactive evaluation report
python evaluate_display.py \
    --results_file evaluation_results/comprehensive_evaluation.json \
    --output_dir evaluation_results

# Open the generated HTML file in your browser
# Location: evaluation_results/reports/evaluation_report_YYYYMMDD_HHMMSS.html
```

### 6. Monitoring with TensorBoard
```bash
# Start TensorBoard server
tensorboard --logdir checkpoints/tensorboard

# Open in browser: http://localhost:6006
```

## Model Customization

### Architecture Parameters
```python
from model import DNABERTModel

# Custom model configuration
model = DNABERTModel(
    sequence_length=2000,           # Longer sequences
    num_classes=5,                  # More annotation types
    hidden_size=512,                # Smaller model
    num_hidden_layers=8,            # Fewer layers
    num_attention_heads=8,          # Fewer attention heads
    classifier_dropout=0.2          # Higher dropout
)
```

### Training Configuration
```python
# Custom training with specific parameters
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=25,
    learning_rate=5e-5,
    weight_decay=0.02,
    max_grad_norm=2.0,
    device="cuda"
)
```

## Examples

### Single Organism Training
```bash
# Fast training on bacterial genome
python model.py \
    --organism "escherichia_coli" \
    --epochs 15 \
    --batch_size 64 \
    --learning_rate 2e-4

# Training on mammalian genome
python model.py \
    --organism "mus_musculus" \
    --epochs 25 \
    --batch_size 32 \
    --sequence_length 1500
```

### Custom Evaluation
```bash
# Detailed evaluation with custom output
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 64 \
    --output_dir detailed_evaluation

# Compare multiple checkpoints
for epoch in 10 15 20; do
    python evaluate.py \
        --checkpoint checkpoints/checkpoint_epoch_${epoch}.pt \
        --output_dir evaluation_epoch_${epoch}
done
```

### Prediction on New Sequences
```python
from model import DNABERTModel

# Load trained model
model = DNABERTModel.from_pretrained("models/dnabert")

# Predict on new sequence
dna_sequence = "ATCGATCGATCG..." * 100  # 1000 bp sequence
predictions = model.predict(dna_sequence)

# Get prediction probabilities
non_coding_prob = predictions[0][0]
coding_prob = predictions[0][1] 
regulatory_prob = predictions[0][2]

print(f"Non-coding: {non_coding_prob:.3f}")
print(f"Coding: {coding_prob:.3f}")
print(f"Regulatory: {regulatory_prob:.3f}")
```

## Performance Optimization

### GPU Optimization
```bash
# Use mixed precision training (faster, less memory)
python model.py --epochs 10 --batch_size 64 --amp

# Optimize for multiple GPUs
python model.py --epochs 10 --batch_size 128 --multi_gpu
```

### Memory Optimization
```bash
# Reduce memory usage
python model.py \
    --batch_size 16 \
    --sequence_length 512 \
    --num_workers 2

# For limited memory systems
python model.py \
    --organism "escherichia_coli" \
    --batch_size 8 \
    --sequence_length 500
```

### Speed Optimization
```bash
# Fast training for development
python model.py \
    --organism "escherichia_coli" \
    --epochs 5 \
    --batch_size 64 \
    --num_workers 8
```

## Troubleshooting Guide

### Training Issues

#### Out of Memory Errors
```bash
# Reduce batch size
python model.py --batch_size 8

# Reduce sequence length
python model.py --sequence_length 512

# Use single organism
python model.py --organism "escherichia_coli"
```

#### Slow Training
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Increase batch size (if memory allows)
python model.py --batch_size 64

# Use more workers
python model.py --num_workers 8
```

#### Poor Performance
```bash
# Increase training epochs
python model.py --epochs 50

# Adjust learning rate
python model.py --learning_rate 5e-5

# Use larger model
python model.py --hidden_size 1024 --num_hidden_layers 16
```

### Evaluation Issues

#### Missing Checkpoints
```bash
# List available checkpoints
ls checkpoints/

# Use latest checkpoint
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_$(ls checkpoints/ | grep -o 'epoch_[0-9]*' | sort -V | tail -1 | cut -d_ -f2).pt
```

#### Report Generation Fails
```bash
# Check results file exists
ls evaluation_results/comprehensive_evaluation.json

# Generate with verbose output
python evaluate_display.py --results_file evaluation_results/comprehensive_evaluation.json --verbose
```

## File Structure

```
genome-annotation-model/
├── model.py                      # Main training script with DNABERTModel
├── evaluate.py                   # Model evaluation script
├── evaluate_display.py           # HTML report generation
├── download_annotations.py       # Data download script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── genome_annotations/           # Downloaded genome data
│   ├── refseq/                  # RefSeq GFF files
│   └── fasta/                   # FASTA sequence files
├── checkpoints/                  # Model checkpoints
│   ├── best_model.pt            # Best performing model
│   ├── checkpoint_epoch_*.pt    # Per-epoch checkpoints
│   └── tensorboard/             # TensorBoard logs
├── models/                       # Final saved models
│   └── dnabert/                 # DNA-BERT model files
├── evaluation_results/           # Evaluation outputs
│   ├── comprehensive_evaluation.json
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── reports/                 # HTML evaluation reports
└── training.log                 # Training log file
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 20GB free disk space
- CPU training support

### Recommended Requirements
- Python 3.9+
- CUDA-capable GPU (RTX 3070 or better)
- 16GB+ RAM
- 50GB+ free disk space
- NVIDIA CUDA 11.0+

### Optimal Requirements
- Python 3.10+
- High-end GPU (RTX 4080/4090, A100)
- 32GB+ RAM
- 100GB+ SSD storage
- NVIDIA CUDA 12.0+

## Citation

If you use this model in your research, please cite:
```bibtex
@software{dnabert_genome_annotation,
  author = {Your Name},
  title = {DNA-BERT Genome Annotation Model},
  year = {2024},
  url = {https://github.com/yourusername/genome-annotation-model},
  note = {Deep learning model for genome annotation using DNA-BERT architecture}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DNA-BERT architecture inspiration
- NCBI and Ensembl for genomic data
- PyTorch and Transformers libraries
- Scientific computing community