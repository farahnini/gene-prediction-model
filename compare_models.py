import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class BaseModel(nn.Module):
    def __init__(self, sequence_length=1000, num_classes=3):
        super(BaseModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes

class CNNLSTMModel(BaseModel):
    def __init__(self, sequence_length=1000, num_classes=3):
        super(CNNLSTMModel, self).__init__(sequence_length, num_classes)
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        out = self.fc(context)
        return out

class TransformerModel(BaseModel):
    def __init__(self, sequence_length=1000, num_classes=3):
        super(TransformerModel, self).__init__(sequence_length, num_classes)
        
        # Embedding layer
        self.embedding = nn.Linear(4, 256)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(256, sequence_length)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x.permute(0, 2, 1))
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        out = self.classifier(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ResNetModel(BaseModel):
    def __init__(self, sequence_length=1000, num_classes=3):
        super(ResNetModel, self).__init__(sequence_length, num_classes)
        
        # Initial convolution
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = nn.ReLU(inplace=True)(out)
        
        return out

class EfficientNetModel(BaseModel):
    def __init__(self, sequence_length=1000, num_classes=3):
        super(EfficientNetModel, self).__init__(sequence_length, num_classes)
        
        # Initial convolution
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConv(32, 16, 1, 1),
            MBConv(16, 24, 6, 2),
            MBConv(24, 40, 6, 2),
            MBConv(40, 80, 6, 2),
            MBConv(80, 112, 6, 1),
            MBConv(112, 192, 6, 2),
            MBConv(192, 320, 6, 1)
        )
        
        # Final layers
        self.conv2 = nn.Conv1d(320, 1280, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(1280)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.blocks(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        
        hidden_channels = int(in_channels * expansion_factor)
        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x += residual
        
        return x

class ModelComparator:
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = os.path.join(results_dir, "models")
        self.comparison_dir = os.path.join(results_dir, "comparison")
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            "CNN-LSTM": CNNLSTMModel(),
            "Transformer": TransformerModel(),
            "ResNet": ResNetModel(),
            "EfficientNet": EfficientNetModel()
        }
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.001
    
    def train_model(self, model, train_loader, val_loader):
        """Train a single model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        best_val_acc = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(self.num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(self.models_dir, f"{model.__class__.__name__}.pth"))
        
        return history
    
    def evaluate_model(self, model, test_loader):
        """Evaluate a single model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1_score': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return metrics
    
    def compare_models(self, train_loader, val_loader, test_loader):
        """Compare all models and generate report."""
        results = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            history = self.train_model(model, train_loader, val_loader)
            
            print(f"Evaluating {name} model...")
            metrics = self.evaluate_model(model, test_loader)
            
            results[name] = {
                'metrics': metrics,
                'history': history
            }
        
        # Generate comparison report
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results):
        """Generate comparison report with visualizations."""
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score')
        )
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(metrics, positions):
            values = [results[model]['metrics'][metric] for model in self.models.keys()]
            fig.add_trace(
                go.Bar(
                    x=list(self.models.keys()),
                    y=values,
                    name=metric.capitalize()
                ),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(
            title="Model Comparison",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = os.path.join(self.comparison_dir, "model_comparison.html")
        fig.write_html(plot_path)
        
        # Save metrics
        metrics_path = os.path.join(self.comparison_dir, "model_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate summary
        summary = "Model Comparison Summary\n"
        summary += "======================\n\n"
        
        for name, result in results.items():
            summary += f"{name} Model:\n"
            summary += f"  Accuracy:  {result['metrics']['accuracy']:.4f}\n"
            summary += f"  Precision: {result['metrics']['precision']:.4f}\n"
            summary += f"  Recall:    {result['metrics']['recall']:.4f}\n"
            summary += f"  F1 Score:  {result['metrics']['f1_score']:.4f}\n\n"
        
        # Save summary
        summary_path = os.path.join(self.comparison_dir, "comparison_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"\nComparison report generated in {self.comparison_dir}")

def main():
    # Example usage
    comparator = ModelComparator()
    
    # Load your data here
    # train_loader = ...
    # val_loader = ...
    # test_loader = ...
    
    # Compare models
    results = comparator.compare_models(train_loader, val_loader, test_loader)
    
    # Print best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    print(f"\nBest performing model: {best_model[0]}")
    print(f"Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    main() 