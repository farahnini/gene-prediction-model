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

## Installation

### Option 1: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda create -n genome-annotation python=3.8
conda activate genome-annotation
pip install -r requirements.txt
```

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