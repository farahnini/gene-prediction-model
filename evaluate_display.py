import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import jinja2
import base64
from io import BytesIO

class EvaluationDisplay:
    def __init__(self, results_dir="results"):
        """
        Initialize the evaluation display generator.
        
        Args:
            results_dir (str): Directory containing evaluation results
        """
        self.results_dir = results_dir
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.output_dir = os.path.join(results_dir, "reports")
        
        # Create necessary directories
        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir)
        )
        
        # Create template if it doesn't exist
        self._create_template()
    
    def _create_template(self):
        """Create the HTML template for the evaluation report."""
        template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genome Annotation Model Evaluation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f8f9fa; }
        .metric-card { 
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            padding: 20px;
            background: white;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 1.2em;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .plot-container {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 0;
            margin-bottom: 30px;
        }
        .section-header {
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin: 30px 0 20px 0;
        }
        .accuracy-high { color: #28a745; }
        .accuracy-medium { color: #ffc107; }
        .accuracy-low { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="display-4">🧬 Genome Annotation Model</h1>
            <h2>Evaluation Report</h2>
            <p class="lead">Generated on {{ timestamp }}</p>
        </div>
    </div>

    <div class="container">
        <!-- Overall Metrics -->
        <h2 class="section-header">📊 Overall Performance Metrics</h2>
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value {{ 'accuracy-high' if metrics.accuracy > 0.8 else ('accuracy-medium' if metrics.accuracy > 0.6 else 'accuracy-low') }}">
                        {{ "%.2f"|format(metrics.accuracy * 100) }}%
                    </div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f"|format(metrics.precision * 100) }}%</div>
                    <div class="metric-label">Precision</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f"|format(metrics.recall * 100) }}%</div>
                    <div class="metric-label">Recall</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f"|format(metrics.f1_score * 100) }}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
        </div>

        <!-- Training History (if available) -->
        {% if training_history_plot %}
        <h2 class="section-header">📈 Training History</h2>
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    {{ training_history_plot }}
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Performance Analysis -->
        <h2 class="section-header">🔍 Performance Analysis</h2>
        <div class="row">
            <div class="col-md-6">
                <div class="plot-container">
                    <h4>Performance by Species/Dataset</h4>
                    {{ species_performance_plot }}
                </div>
            </div>
            <div class="col-md-6">
                <div class="plot-container">
                    <h4>Confusion Matrix</h4>
                    {{ confusion_matrix_plot }}
                </div>
            </div>
        </div>

        <!-- Per-Class Performance -->
        {% if per_class_plot %}
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    <h4>Per-Class Performance Metrics</h4>
                    {{ per_class_plot }}
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Detailed Results Table -->
        <h2 class="section-header">📋 Detailed Results</h2>
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    <h4>Results by Species/Dataset</h4>
                    <div class="table-responsive">
                        {{ species_table }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Per-Class Results Table -->
        {% if per_class_table %}
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    <h4>Per-Class Performance</h4>
                    <div class="table-responsive">
                        {{ per_class_table }}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Error Analysis -->
        <h2 class="section-header">⚠️ Error Analysis</h2>
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    {{ error_analysis_plot }}
                </div>
            </div>
        </div>

        <!-- Model Information -->
        <h2 class="section-header">ℹ️ Model Information</h2>
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    <div class="row">
                        <div class="col-md-6">
                            <h5>Training Configuration</h5>
                            <ul>
                                <li><strong>Model:</strong> DNA-BERT with CNN-LSTM</li>
                                <li><strong>Total Samples:</strong> {{ metrics.total_samples if metrics.total_samples else 'N/A' }}</li>
                                <li><strong>Classes:</strong> Non-coding, Coding, Regulatory</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h5>Performance Summary</h5>
                            <ul>
                                <li><strong>Best Accuracy:</strong> {{ "%.2f"|format(metrics.accuracy * 100) }}%</li>
                                <li><strong>Model Type:</strong> Multi-class Classification</li>
                                <li><strong>Evaluation Date:</strong> {{ timestamp.split(' ')[0] }}</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="mt-5 py-4 bg-dark text-white text-center">
        <div class="container">
            <p>&copy; {{ timestamp.split(' ')[0] }} Genome Annotation Model - Evaluation Report</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        template_path = os.path.join(self.template_dir, "evaluation_report.html")
        with open(template_path, "w") as f:
            f.write(template)
    
    def _create_training_history_plot(self, history):
        """Create training history plots."""
        if not history:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'Epoch Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', 
                      line=dict(color='blue')), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                      line=dict(color='red')), row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                      line=dict(color='blue'), showlegend=False), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                      line=dict(color='red'), showlegend=False), row=1, col=2
        )
        
        # Learning rate plot (if available)
        if 'learning_rates' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['learning_rates'], name='Learning Rate',
                          line=dict(color='green'), showlegend=False), row=2, col=1
            )
            fig.update_yaxes(type="log", row=2, col=1)
        
        # Epoch time plot (if available)
        if 'epoch_times' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['epoch_times'], name='Epoch Time',
                          line=dict(color='purple'), showlegend=False), row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Training History")
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_yaxes(title_text="Time (s)", row=2, col=2)
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_per_class_plot(self, per_class_metrics):
        """Create per-class performance plot."""
        if not per_class_metrics:
            return None
        
        classes = list(per_class_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        for metric in metrics:
            values = [per_class_metrics[cls][metric] for cls in classes]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=classes,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Per-Class Performance Metrics",
            xaxis_title="Class",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_per_class_table(self, per_class_metrics):
        """Create per-class performance table."""
        if not per_class_metrics:
            return None
        
        df = pd.DataFrame.from_dict(per_class_metrics, orient='index')
        df = df.round(4)
        df.index.name = 'Class'
        return df.to_html(classes='table table-striped table-hover')

    def _create_species_performance_plot(self, results):
        """Create a bar plot showing performance metrics by species."""
        species = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        for metric in metrics:
            values = [results[sp][metric] for sp in species]
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=species,
                y=values,
                text=[f"{v:.2%}" for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Performance Metrics by Species",
            xaxis_title="Species",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(tickformat='.0%'),
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_confusion_matrix_plot(self, confusion_matrix):
        """Create a heatmap of the confusion matrix."""
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='RdBu',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_error_analysis_plot(self, results):
        """Create a plot showing error distribution and patterns."""
        species = list(results.keys())
        error_rates = [1 - results[sp]['accuracy'] for sp in species]
        
        fig = go.Figure(data=go.Bar(
            x=species,
            y=error_rates,
            text=[f"{v:.2%}" for v in error_rates],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Error Rates by Species",
            xaxis_title="Species",
            yaxis_title="Error Rate",
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_species_table(self, results):
        """Create an HTML table with detailed results by species."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.round(4)
        df.index.name = 'Species'
        return df.to_html(classes='table table-striped table-hover')
    
    def generate_report(self, evaluation_results):
        """
        Generate an HTML evaluation report.
        
        Args:
            evaluation_results (dict): Dictionary containing evaluation results
        """
        # Extract metrics
        metrics = evaluation_results['overall_metrics']
        results = evaluation_results['species_results']
        confusion_matrix = evaluation_results['confusion_matrix']
        per_class_metrics = evaluation_results.get('per_class_metrics', {})
        training_history = evaluation_results.get('training_history', {})
        
        # Create plots
        species_performance_plot = self._create_species_performance_plot(results)
        confusion_matrix_plot = self._create_confusion_matrix_plot(confusion_matrix)
        error_analysis_plot = self._create_error_analysis_plot(results)
        species_table = self._create_species_table(results)
        
        # Create additional plots for enhanced report
        training_history_plot = self._create_training_history_plot(training_history)
        per_class_plot = self._create_per_class_plot(per_class_metrics)
        per_class_table = self._create_per_class_table(per_class_metrics)
        
        # Render template
        template = self.env.get_template("evaluation_report.html")
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics,
            species_performance_plot=species_performance_plot,
            confusion_matrix_plot=confusion_matrix_plot,
            error_analysis_plot=error_analysis_plot,
            species_table=species_table,
            training_history_plot=training_history_plot,
            per_class_plot=per_class_plot,
            per_class_table=per_class_table
        )
        
        # Save report
        report_path = os.path.join(
            self.output_dir,
            f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(report_path, "w") as f:
            f.write(html_content)
        
        return report_path

def main():
    """Example usage with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Evaluation Report')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to comprehensive evaluation JSON file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Load evaluation results
    with open(args.results_file, 'r') as f:
        evaluation_results = json.load(f)
    
    # Generate report
    display = EvaluationDisplay(results_dir=args.output_dir)
    report_path = display.generate_report(evaluation_results)
    print(f"Evaluation report generated: {report_path}")

if __name__ == "__main__":
    main()