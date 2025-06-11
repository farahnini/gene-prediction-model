# Genome Annotation Model

A deep learning model for genome annotation, trained on diverse species to predict functional elements in DNA sequences.

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

# Train the model
python model.py

# Evaluate the model
python evaluate.py

# View evaluation results
# Open the generated HTML report in your browser
```

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

The model uses a hybrid architecture combining:
- Convolutional layers for local pattern recognition
- Bidirectional LSTM layers for sequence context
- Attention mechanism for focusing on relevant regions
- Dense layers for final classification

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

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB RAM minimum
- 50GB free disk space

## Usage

### 1. Download Genome Annotations
```bash
python download_annotations.py
```
Edit the email in `download_annotations.py` before running.

### 2. Train the Model
```bash
python model.py
```
The model will be saved in the `models` directory.

### 3. Evaluate the Model
```bash
python evaluate.py
```
This generates an HTML report in the `results/reports` directory.

### 4. Compare Models
```bash
python compare_models.py
```
This will train and evaluate different model architectures.

## Model Customization

### Architecture Modifications
```python
from model import GenomeAnnotationModel

# Add convolutional layers
model = GenomeAnnotationModel(
    num_conv_layers=4,
    conv_filters=[64, 128, 256, 512]
)
```

### Hyperparameter Tuning
```python
# Custom training parameters
model.train(
    batch_size=32,
    learning_rate=0.001,
    epochs=50
)
```

## Examples

### Training with Custom Parameters
```python
from model import GenomeAnnotationModel

model = GenomeAnnotationModel(
    sequence_length=1000,
    num_classes=5,
    learning_rate=0.001
)

model.train(
    batch_size=32,
    epochs=50,
    validation_split=0.2
)
```

### Processing Custom Genome Data
```python
from model import GenomeAnnotationModel

model = GenomeAnnotationModel()
model.process_sequence("ATCG...")  # Your DNA sequence
```

### Evaluating on New Species
```python
from model import GenomeAnnotationModel

model = GenomeAnnotationModel()
model.load_weights("models/best_model.h5")
results = model.evaluate("path/to/new/species.fasta")
```

## Troubleshooting Guide

### Download Issues
- **Problem**: NCBI connection timeout
  - **Solution**: Check internet connection and try again
- **Problem**: Ensembl download fails
  - **Solution**: Verify species name format

### Training Issues
- **Problem**: Out of memory
  - **Solution**: Reduce batch size or sequence length
- **Problem**: Slow training
  - **Solution**: Enable GPU acceleration

### Evaluation Issues
- **Problem**: Missing metrics
  - **Solution**: Check data format and labels
- **Problem**: Low accuracy
  - **Solution**: Verify data preprocessing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

If you use this model in your research, please cite:
```
@software{genome_annotation_model,
  author = {Your Name},
  title = {Genome Annotation Model},
  year = {2024},
  url = {https://github.com/yourusername/genome-annotation-model}
}
``` 