import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import GenomeAnnotationModel, GenomeDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        tuple: (predictions, true_labels, accuracy)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Evaluating"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return np.array(all_predictions), np.array(all_labels), accuracy

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_metrics(metrics_df):
    """
    Plot training metrics.
    
    Args:
        metrics_df: DataFrame containing training metrics
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train Accuracy')
    plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    # Note: You'll need to implement the test data loading logic
    test_sequences = None  # Load your test sequences
    test_annotations = None  # Load your test annotations
    
    test_dataset = GenomeDataset(test_sequences, test_annotations)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load trained model
    model = GenomeAnnotationModel()
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    
    # Evaluate model
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    
    # Print overall accuracy
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Generate classification report
    class_names = ['Non-coding', 'Coding', 'Regulatory']  # Update with your actual class names
    report = classification_report(true_labels, predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names)
    
    # Load and plot training metrics
    try:
        metrics_df = pd.read_csv('training_metrics.csv')
        plot_metrics(metrics_df)
    except FileNotFoundError:
        print("Training metrics file not found. Skipping metrics plot.")

if __name__ == "__main__":
    main() 