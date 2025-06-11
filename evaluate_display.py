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
        body { font-family: Arial, sans-serif; }
        .metric-card { 
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            padding: 20px;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 1.2em;
            color: #7f8c8d;
        }
        .plot-container {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .species-table {
            margin: 20px 0;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>Genome Annotation Model Evaluation Report</h1>
            <p class="lead">Generated on {{ timestamp }}</p>
        </div>
    </div>

    <div class="container">
        <!-- Overall Metrics -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card bg-light">
                    <div class="metric-value">{{ "%.2f"|format(metrics.accuracy * 100) }}%</div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card bg-light">
                    <div class="metric-value">{{ "%.2f"|format(metrics.precision * 100) }}%</div>
                    <div class="metric-label">Precision</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card bg-light">
                    <div class="metric-value">{{ "%.2f"|format(metrics.recall * 100) }}%</div>
                    <div class="metric-label">Recall</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card bg-light">
                    <div class="metric-value">{{ "%.2f"|format(metrics.f1_score * 100) }}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
        </div>

        <!-- Performance Plots -->
        <div class="row">
            <div class="col-md-6">
                <div class="plot-container">
                    <h3>Performance by Species</h3>
                    {{ species_performance_plot }}
                </div>
            </div>
            <div class="col-md-6">
                <div class="plot-container">
                    <h3>Confusion Matrix</h3>
                    {{ confusion_matrix_plot }}
                </div>
            </div>
        </div>

        <!-- Detailed Results -->
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    <h3>Detailed Results by Species</h3>
                    <div class="table-responsive">
                        {{ species_table }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Error Analysis -->
        <div class="row">
            <div class="col-12">
                <div class="plot-container">
                    <h3>Error Analysis</h3>
                    {{ error_analysis_plot }}
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        template_path = os.path.join(self.template_dir, "evaluation_report.html")
        with open(template_path, "w") as f:
            f.write(template)
    
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
        
        # Create plots
        species_performance_plot = self._create_species_performance_plot(results)
        confusion_matrix_plot = self._create_confusion_matrix_plot(confusion_matrix)
        error_analysis_plot = self._create_error_analysis_plot(results)
        species_table = self._create_species_table(results)
        
        # Render template
        template = self.env.get_template("evaluation_report.html")
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics,
            species_performance_plot=species_performance_plot,
            confusion_matrix_plot=confusion_matrix_plot,
            error_analysis_plot=error_analysis_plot,
            species_table=species_table
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
    # Example usage
    results_dir = "results"
    display = EvaluationDisplay(results_dir)
    
    # Example evaluation results
    evaluation_results = {
        'overall_metrics': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85
        },
        'species_results': {
            'Homo sapiens': {
                'accuracy': 0.88,
                'precision': 0.86,
                'recall': 0.90,
                'f1_score': 0.88
            },
            'Mus musculus': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84,
                'f1_score': 0.82
            }
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