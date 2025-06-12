import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from model import DNABERTModel, GenomeDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import os
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate the model on test data with detailed metrics.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: List of class names
    
    Returns:
        dict: Comprehensive evaluation results
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct = 0
    total = 0
    
    logging.info("Starting model evaluation...")
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
        
        for batch_idx, batch in enumerate(eval_pbar):
            sequences = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            
            # Convert sequences to token format for BERT
            if len(sequences.shape) == 3:  # one-hot encoded
                sequences = torch.argmax(sequences, dim=-1)
            
            # Convert multi-label to single label
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels_single = torch.argmax(labels, dim=1)
            else:
                labels_single = labels.squeeze()
            
            outputs = model(sequences)
            probabilities = torch.softmax(outputs[0], dim=-1)
            _, predicted = outputs[0].max(1)
            
            total += labels_single.size(0)
            correct += predicted.eq(labels_single).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_single.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            current_acc = correct / total
            eval_pbar.set_postfix({
                'Accuracy': f'{current_acc:.4f}',
                'Samples': f'{total}'
            })
    
    accuracy = correct / total
    
    # Calculate additional metrics
    predictions_array = np.array(all_predictions)
    labels_array = np.array(all_labels)
    probabilities_array = np.array(all_probabilities)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array, predictions_array, average=None, labels=range(len(class_names))
    )
    
    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_array, predictions_array, average='macro'
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels_array, predictions_array)
    
    # Compile results
    results = {
        'overall_metrics': {
            'accuracy': accuracy,
            'precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro,
            'total_samples': total
        },
        'per_class_metrics': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            } for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'predictions': predictions_array.tolist(),
        'true_labels': labels_array.tolist(),
        'probabilities': probabilities_array.tolist(),
        'class_names': class_names
    }
    
    logging.info(f"Evaluation completed - Accuracy: {accuracy:.4f}")
    return results

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved to {save_path}")

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training history from checkpoint.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    if not history:
        logging.warning("No training history provided")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    if 'learning_rates' in history:
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Plot epoch times
    if 'epoch_times' in history:
        axes[1, 1].plot(epochs, history['epoch_times'], 'm-')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training history plot saved to {save_path}")

def plot_class_distribution(labels, class_names, save_path="class_distribution.png"):
    """
    Plot class distribution in the dataset.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Class Distribution in Test Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Class distribution plot saved to {save_path}")

def save_evaluation_results(results, save_path="evaluation_results.json"):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        save_path: Path to save the results
    """
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Evaluation results saved to {save_path}")

def load_checkpoint(checkpoint_path):
    """
    Load model checkpoint and extract training history.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        tuple: (model_state_dict, history)
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    history = checkpoint.get('history', {})
    model_state_dict = checkpoint['model_state_dict']
    
    logging.info(f"Checkpoint loaded - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    return model_state_dict, history

def main():
    parser = argparse.ArgumentParser(description='Evaluate Genome Annotation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='genome_annotations',
                       help='Directory containing test data')
    parser.add_argument('--organism', type=str, default=None,
                       help='Single organism to evaluate (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--sequence_length', type=int, default=1000,
                       help='Sequence length')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Define class names
    class_names = ['Non-coding', 'Coding', 'Regulatory']
    
    # Load test data (you'll need to implement this based on your data structure)
    logging.info("Loading test data...")
    try:
        from model import load_and_preprocess_data
        from sklearn.model_selection import train_test_split
        
        # Load data
        sequences, annotations = load_and_preprocess_data(
            args.data_dir, 
            sequence_length=args.sequence_length,
            single_organism=args.organism
        )
        
        # Split to get test set (using same random state as training)
        _, test_sequences, _, test_annotations = train_test_split(
            sequences, annotations, test_size=0.2, random_state=42, 
            stratify=annotations.argmax(axis=1)
        )
        
        logging.info(f"Test dataset size: {len(test_sequences)} samples")
        
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return
    
    # Create test dataset and dataloader
    test_dataset = GenomeDataset(test_sequences, test_annotations)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Load model
    logging.info("Loading model...")
    model = DNABERTModel(
        sequence_length=args.sequence_length,
        num_classes=len(class_names)
    )
    
    # Load checkpoint
    model_state_dict, training_history = load_checkpoint(args.checkpoint)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    
    logging.info("Model loaded successfully")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Print results
    logging.info("\n" + "="*60)
    logging.info("EVALUATION RESULTS")
    logging.info("="*60)
    logging.info(f"Overall Accuracy: {results['overall_metrics']['accuracy']:.4f}")
    logging.info(f"Overall Precision: {results['overall_metrics']['precision']:.4f}")
    logging.info(f"Overall Recall: {results['overall_metrics']['recall']:.4f}")
    logging.info(f"Overall F1-Score: {results['overall_metrics']['f1_score']:.4f}")
    
    logging.info("\nPer-Class Results:")
    for class_name in class_names:
        metrics = results['per_class_metrics'][class_name]
        logging.info(f"{class_name}:")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logging.info(f"  Support: {metrics['support']}")
    
    # Generate and save plots
    logging.info("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(np.array(results['confusion_matrix']), class_names, cm_path)
    
    # Training history
    if training_history:
        history_path = os.path.join(args.output_dir, "training_history.png")
        plot_training_history(training_history, history_path)
    
    # Class distribution
    dist_path = os.path.join(args.output_dir, "class_distribution.png")
    plot_class_distribution(results['true_labels'], class_names, dist_path)
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    save_evaluation_results(results, results_path)
    
    # Create species results for compatibility with evaluation display
    species_results = {
        args.organism if args.organism else 'All_Species': {
            'accuracy': results['overall_metrics']['accuracy'],
            'precision': results['overall_metrics']['precision'],
            'recall': results['overall_metrics']['recall'],
            'f1_score': results['overall_metrics']['f1_score']
        }
    }
    
    # Add training history if available
    evaluation_data = {
        'overall_metrics': results['overall_metrics'],
        'species_results': species_results,
        'confusion_matrix': results['confusion_matrix'],
        'per_class_metrics': results['per_class_metrics'],
        'training_history': training_history
    }
    
    # Save comprehensive evaluation data
    comprehensive_path = os.path.join(args.output_dir, "comprehensive_evaluation.json")
    with open(comprehensive_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    logging.info(f"\nEvaluation completed! Results saved to {args.output_dir}")
    logging.info(f"Use the comprehensive_evaluation.json with evaluate_display.py to generate HTML report")

if __name__ == "__main__":
    main()