"""
Evaluation metrics for EAGLE framework.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class EAGLEMetrics:
    """
    Comprehensive metrics evaluation for EAGLE model.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        probabilities: torch.Tensor
    ):
        """
        Update metrics with new batch predictions.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities
        """
        # Convert to numpy if tensors
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.labels.extend(labels.flatten())
        self.probabilities.extend(probabilities)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        probabilities = np.array(self.probabilities)
        
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        # AUROC (using probabilities for positive class)
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            pos_probs = probabilities[:, 1]  # Probability of positive class
        else:
            pos_probs = probabilities
        
        auroc = roc_auc_score(labels, pos_probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'auroc': auroc,
            'precision_human': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
            'precision_llm': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
            'recall_human': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
            'recall_llm': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
            'f1_human': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
            'f1_llm': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions:
            return np.array([])
        
        return confusion_matrix(self.labels, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.predictions:
            return ""
        
        class_names = ['Human', 'LLM-Generated']
        return classification_report(
            self.labels, self.predictions, 
            target_names=class_names
        )
    
    def plot_confusion_matrix(self, save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Human', 'LLM-Generated'],
            yticklabels=['Human', 'LLM-Generated'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('EAGLE Detection Confusion Matrix')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def evaluate_cross_domain(
    model, 
    source_loader, 
    target_loader, 
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate cross-domain performance.
    
    Args:
        model: Trained EAGLE model
        source_loader: Source domain data loader
        target_loader: Target domain data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with source and target domain metrics
    """
    model.eval()
    
    results = {}
    
    for domain, loader in [('source', source_loader), ('target', target_loader)]:
        metrics = EAGLEMetrics()
        
        with torch.no_grad():
            for batch in loader:
                images = batch['images'].to(device)
                texts = batch['texts']
                labels = batch['labels'].to(device)
                
                outputs = model(images, texts)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                probabilities = torch.softmax(outputs['logits'], dim=-1)
                
                metrics.update(predictions, labels, probabilities)
        
        results[domain] = metrics.compute()
    
    return results


def compute_emotional_authenticity_gap_stats(
    human_texts: List[str], 
    llm_texts: List[str],
    emotion_classifier
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical analysis of emotional authenticity gaps.
    
    Args:
        human_texts: List of human-authored texts
        llm_texts: List of LLM-generated texts
        emotion_classifier: Trained emotion classifier
        
    Returns:
        Statistical comparison of emotional patterns
    """
    emotion_classifier.eval()
    
    # Extract emotion features
    with torch.no_grad():
        human_emotions = emotion_classifier(human_texts).cpu().numpy()
        llm_emotions = emotion_classifier(llm_texts).cpu().numpy()
    
    emotion_labels = emotion_classifier.emotion_labels
    
    stats = {}
    
    for i, emotion in enumerate(emotion_labels):
        human_scores = human_emotions[:, i]
        llm_scores = llm_emotions[:, i]
        
        stats[emotion] = {
            'human_mean': float(np.mean(human_scores)),
            'human_std': float(np.std(human_scores)),
            'human_var': float(np.var(human_scores)),
            'human_p90': float(np.percentile(human_scores, 90)),
            'llm_mean': float(np.mean(llm_scores)),
            'llm_std': float(np.std(llm_scores)),
            'llm_var': float(np.var(llm_scores)),
            'llm_p90': float(np.percentile(llm_scores, 90)),
            'mean_diff': float(np.mean(human_scores) - np.mean(llm_scores)),
            'var_ratio': float(np.var(human_scores) / (np.var(llm_scores) + 1e-8))
        }
    
    return stats
