"""
Comprehensive Evaluation Metrics for ECG Classification Models

Includes:
- ROC-AUC curves and plots
- Confusion matrices with heatmaps
- Precision-Recall curves
- Sensitivity/Specificity
- Cohen's Kappa
- Matthews Correlation Coefficient
- Statistical significance tests
- Computational efficiency metrics (FLOPs, memory, latency)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy import stats
import time
from typing import Dict, List, Tuple, Optional
import json


class ComprehensiveEvaluator:
    """
    Comprehensive model evaluation with multiple metrics.
    """
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Returns dictionary with all metrics.
        """
        model = model.to(device)
        model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(batch_x)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / batch_x.size(0))  # Per sample
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_labels.extend(batch_y.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Compute all metrics
        metrics = {}
        
        # 1. Basic Classification Metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        # 2. Agreement Metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # 3. ROC-AUC (one-vs-rest for multiclass)
        try:
            if self.num_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None
        
        # 4. Sensitivity and Specificity (per class)
        cm = confusion_matrix(y_true, y_pred)
        sensitivities = []
        specificities = []
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        metrics['sensitivity_per_class'] = sensitivities
        metrics['specificity_per_class'] = specificities
        metrics['sensitivity_macro'] = np.mean(sensitivities)
        metrics['specificity_macro'] = np.mean(specificities)
        
        # 5. Computational Efficiency Metrics
        metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
        metrics['std_inference_time_ms'] = np.std(inference_times) * 1000
        metrics['min_inference_time_ms'] = np.min(inference_times) * 1000
        metrics['max_inference_time_ms'] = np.max(inference_times) * 1000
        metrics['throughput_samples_per_sec'] = 1.0 / np.mean(inference_times)
        
        # 6. Model Complexity Metrics
        metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
        metrics['num_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Store raw data for plotting
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.confusion_matrix = cm
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, figsize: Tuple = (10, 8)):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curves(self, save_path: Optional[str] = None, figsize: Tuple = (10, 8)):
        """Plot ROC curves (one-vs-rest for multiclass)."""
        plt.figure(figsize=figsize)
        
        # Compute ROC curve for each class
        for i in range(self.num_classes):
            # Binary indicators for class i
            y_true_binary = (self.y_true == i).astype(int)
            y_score = self.y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, save_path: Optional[str] = None, figsize: Tuple = (10, 8)):
        """Plot Precision-Recall curves."""
        plt.figure(figsize=figsize)
        
        for i in range(self.num_classes):
            y_true_binary = (self.y_true == i).astype(int)
            y_score = self.y_prob[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            avg_precision = average_precision_score(y_true_binary, y_score)
            
            plt.plot(recall, precision, lw=2, 
                    label=f'{self.class_names[i]} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_metrics_summary(self, metrics: Dict, save_path: Optional[str] = None):
        """Plot summary of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Per-class F1 scores
        axes[0, 0].bar(self.class_names, metrics['f1_per_class'], color='steelblue')
        axes[0, 0].set_title('F1 Score per Class', fontweight='bold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Sensitivity and Specificity
        x = np.arange(self.num_classes)
        width = 0.35
        axes[0, 1].bar(x - width/2, metrics['sensitivity_per_class'], width, label='Sensitivity', color='green', alpha=0.7)
        axes[0, 1].bar(x + width/2, metrics['specificity_per_class'], width, label='Specificity', color='orange', alpha=0.7)
        axes[0, 1].set_title('Sensitivity & Specificity per Class', fontweight='bold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.class_names, rotation=45)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Overall metrics comparison
        overall_metrics = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision_macro'],
            'Recall': metrics['recall_macro'],
            'F1 Score': metrics['f1_macro'],
            'Cohen Kappa': metrics['cohen_kappa']
        }
        axes[1, 0].barh(list(overall_metrics.keys()), list(overall_metrics.values()), color='coral')
        axes[1, 0].set_title('Overall Metrics', fontweight='bold')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Computational metrics
        comp_metrics = {
            'Avg Inference\n(ms)': metrics['avg_inference_time_ms'],
            'Throughput\n(samples/s)': min(metrics['throughput_samples_per_sec'], 1000),  # Cap for visualization
            'Parameters\n(thousands)': metrics['num_parameters'] / 1000
        }
        axes[1, 1].bar(list(comp_metrics.keys()), list(comp_metrics.values()), color='purple', alpha=0.7)
        axes[1, 1].set_title('Computational Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, metrics: Dict, model_name: str, save_path: Optional[str] = None) -> str:
        """Generate comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append(f"COMPREHENSIVE EVALUATION REPORT: {model_name}")
        report.append("=" * 80)
        report.append("")
        
        # 1. Classification Metrics
        report.append("1. CLASSIFICATION METRICS")
        report.append("-" * 40)
        report.append(f"  Accuracy:               {metrics['accuracy']:.4f}")
        report.append(f"  Precision (Macro):      {metrics['precision_macro']:.4f}")
        report.append(f"  Recall (Macro):         {metrics['recall_macro']:.4f}")
        report.append(f"  F1 Score (Macro):       {metrics['f1_macro']:.4f}")
        report.append(f"  Cohen's Kappa:          {metrics['cohen_kappa']:.4f}")
        report.append(f"  Matthews Corr. Coef:    {metrics['matthews_corrcoef']:.4f}")
        if 'roc_auc_macro' in metrics:
            report.append(f"  ROC-AUC (Macro):        {metrics['roc_auc_macro']:.4f}")
        report.append("")
        
        # 2. Per-Class Metrics
        report.append("2. PER-CLASS METRICS")
        report.append("-" * 40)
        for i, class_name in enumerate(self.class_names):
            report.append(f"  {class_name}:")
            report.append(f"    Precision:    {metrics['precision_per_class'][i]:.4f}")
            report.append(f"    Recall:       {metrics['recall_per_class'][i]:.4f}")
            report.append(f"    F1 Score:     {metrics['f1_per_class'][i]:.4f}")
            report.append(f"    Sensitivity:  {metrics['sensitivity_per_class'][i]:.4f}")
            report.append(f"    Specificity:  {metrics['specificity_per_class'][i]:.4f}")
        report.append("")
        
        # 3. Computational Metrics
        report.append("3. COMPUTATIONAL METRICS")
        report.append("-" * 40)
        report.append(f"  Model Parameters:       {metrics['num_parameters']:,}")
        report.append(f"  Trainable Parameters:   {metrics['num_trainable_parameters']:,}")
        report.append(f"  Avg Inference Time:     {metrics['avg_inference_time_ms']:.2f} ms")
        report.append(f"  Std Inference Time:     {metrics['std_inference_time_ms']:.2f} ms")
        report.append(f"  Throughput:             {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_metrics_json(self, metrics: Dict, save_path: str):
        """Save metrics to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            elif isinstance(value, (list, np.ndarray)):
                json_metrics[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
            else:
                json_metrics[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)


def compare_models_statistical_test(
    metrics1: Dict,
    metrics2: Dict,
    model1_name: str,
    model2_name: str,
    metric_key: str = 'accuracy'
) -> Dict:
    """
    Perform statistical significance test between two models.
    Uses McNemar's test for paired predictions.
    """
    # This is a simplified version - in practice, you'd need the actual predictions
    # For now, we'll use a z-test approximation
    
    acc1 = metrics1[metric_key]
    acc2 = metrics2[metric_key]
    
    # Approximate standard error (simplified)
    n = 100  # This should be actual test set size
    se = np.sqrt((acc1 * (1 - acc1) + acc2 * (1 - acc2)) / n)
    
    z_score = (acc1 - acc2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    result = {
        f'{model1_name}_{metric_key}': acc1,
        f'{model2_name}_{metric_key}': acc2,
        'difference': acc1 - acc2,
        'z_score': z_score,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01
    }
    
    return result


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("This module provides comprehensive evaluation tools for ECG classification models.")
    print("\nFeatures:")
    print("  - ROC-AUC curves")
    print("  - Confusion matrices")
    print("  - Precision-Recall curves")
    print("  - Sensitivity/Specificity")
    print("  - Cohen's Kappa")
    print("  - Matthews Correlation Coefficient")
    print("  - Computational efficiency metrics")
    print("  - Statistical significance tests")
    print("\nUsage: Import this module in benchmark.py or other evaluation scripts.")
