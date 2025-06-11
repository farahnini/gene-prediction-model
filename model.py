import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from evaluate_display import EvaluationDisplay

class GenomeDataset(Dataset):
    def __init__(self, sequences, annotations, sequence_length=1000):
        self.sequences = sequences
        self.annotations = annotations
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        annotation = self.annotations[idx]
        
        # Convert sequence to one-hot encoding
        sequence_encoded = self._one_hot_encode(sequence)
        
        return torch.FloatTensor(sequence_encoded), torch.FloatTensor(annotation)
    
    def _one_hot_encode(self, sequence):
        # Convert DNA sequence to one-hot encoding
        mapping = {'A': [1, 0, 0, 0],
                  'C': [0, 1, 0, 0],
                  'G': [0, 0, 1, 0],
                  'T': [0, 0, 0, 1],
                  'N': [0, 0, 0, 0]}
        
        encoded = []
        for base in sequence:
            encoded.append(mapping.get(base, [0, 0, 0, 0]))
        return np.array(encoded)

class GenomeAnnotationModel(nn.Module):
    def __init__(self, sequence_length=1000, num_classes=3):
        super(GenomeAnnotationModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(4, 64, kernel_size=8, stride=1, padding=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = sequence_length // 4  # After 3 pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input shape: (batch_size, 4, sequence_length)
        
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def load_and_preprocess_data(data_dir, sequence_length=1000):
    """
    Load and preprocess genome sequences and annotations.
    
    Args:
        data_dir (str): Directory containing genome annotations
        sequence_length (int): Length of sequences to process
    
    Returns:
        tuple: (sequences, annotations)
    """
    sequences = []
    annotations = []
    
    # Process each species directory
    for species_dir in os.listdir(data_dir):
        species_path = os.path.join(data_dir, species_dir)
        if not os.path.isdir(species_path):
            continue
            
        # Process GFF files
        for gff_file in os.listdir(species_path):
            if not gff_file.endswith('.gff'):
                continue
                
            # Load and process GFF file
            gff_path = os.path.join(species_path, gff_file)
            # Add your GFF processing logic here
            # This is a placeholder for the actual implementation
            
    return np.array(sequences), np.array(annotations)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """
    Train the genome annotation model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # TensorBoard writer
    writer = SummaryWriter('runs/genome_annotation')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (sequences, annotations) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            sequences, annotations = sequences.to(device), annotations.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, annotations)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += annotations.size(0)
            train_correct += predicted.eq(annotations).sum().item()
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, annotations in val_loader:
                sequences, annotations = sequences.to(device), annotations.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, annotations)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += annotations.size(0)
                val_correct += predicted.eq(annotations).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    writer.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    sequences, annotations = load_and_preprocess_data('genome_annotations')
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, annotations, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = GenomeDataset(X_train, y_train)
    val_dataset = GenomeDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = GenomeAnnotationModel()
    
    # Train model
    train_model(model, train_loader, val_loader)

    # Create display instance
    display = EvaluationDisplay(results_dir="results")

    # Your evaluation results
    evaluation_results = {
        'overall_metrics': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85
        },
        'species_results': {
            'Species1': {
                'accuracy': 0.88,
                'precision': 0.86,
                'recall': 0.90,
                'f1_score': 0.88
            },
            # ... more species
        },
        'confusion_matrix': [
            [150, 25],
            [30, 145]
        ]
    }

    # Generate report
    report_path = display.generate_report(evaluation_results)
    print(f"Evaluation report generated: {report_path}")

if __name__ == "__main__":
    main() 