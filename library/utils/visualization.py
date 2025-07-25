import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict
import numpy as np

class DriftVisualizer:
    """Visualization utilities for drift detection results"""
    
    def __init__(self):
        self.style = 'seaborn-v0_8-darkgrid'
        self.colors = {
            'kl': '#FF6B6B',
            'hellinger': '#4ECDC4',
            'hybrid': '#45B7D1',
            'threshold': '#FFA07A',
            'drift': '#FF4757'
        }
        plt.style.use(self.style)
    
    def plot_drift_scores(self, kl_scores: List[float], hellinger_scores: List[float], 
                         hybrid_scores: List[float], threshold: float, drift_points: List[int] = None):
        """Plot drift scores over time"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        batches = range(len(kl_scores))
        
        # KL Divergence
        ax1.plot(batches, kl_scores, color=self.colors['kl'], linewidth=2, label='KL Divergence')
        ax1.set_ylabel('KL Score', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Hellinger Distance
        ax2.plot(batches, hellinger_scores, color=self.colors['hellinger'], linewidth=2, label='Hellinger Distance')
        ax2.set_ylabel('Hellinger Score', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Hybrid Score with threshold
        ax3.plot(batches, hybrid_scores, color=self.colors['hybrid'], linewidth=2, label='Hybrid Score')
        ax3.axhline(y=threshold, color=self.colors['threshold'], linestyle='--', linewidth=2, label='Threshold')
        ax3.set_ylabel('Hybrid Score', fontsize=12)
        ax3.set_xlabel('Batch Number', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Mark drift points
        if drift_points:
            for point in drift_points:
                if point < len(kl_scores):
                    ax1.axvline(x=point, color=self.colors['drift'], linestyle=':', alpha=0.6)
                    ax2.axvline(x=point, color=self.colors['drift'], linestyle=':', alpha=0.6)
                    ax3.axvline(x=point, color=self.colors['drift'], linestyle=':', alpha=0.6)
        
        plt.suptitle('Drift Detection Scores Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_accuracy_comparison(self, accuracy_before: List[float], accuracy_after: List[float]):
        """Plot accuracy comparison before and after drift adaptation"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(accuracy_before))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], accuracy_before, width, 
                       label='Before Adaptation', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], accuracy_after, width, 
                       label='After Adaptation', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Drift Event', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy: Before vs After Adaptation', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_drift_summary(self, drift_summary: Dict):
        """Plot drift type detection summary"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Drift type counts
        drift_types = list(drift_summary.keys())
        counts = [drift_summary[dt]["detected"] for dt in drift_types]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax1.bar(drift_types, counts, color=colors, alpha=0.8)
        ax1.set_xlabel('Drift Type', fontsize=12)
        ax1.set_ylabel('Detection Count', fontsize=12)
        ax1.set_title('Drift Type Detection Summary', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        
        # Average scores by drift type
        avg_scores = []
        for dt in drift_types:
            if drift_summary[dt]["detected"] > 0 and drift_summary[dt]["hybrid_score"]:
                avg_scores.append(np.mean(drift_summary[dt]["hybrid_score"]))
            else:
                avg_scores.append(0)
        
        bars2 = ax2.bar(drift_types, avg_scores, color=colors, alpha=0.8)
        ax2.set_xlabel('Drift Type', fontsize=12)
        ax2.set_ylabel('Average Hybrid Score', fontsize=12)
        ax2.set_title('Average Detection Score by Drift Type', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_performance_metrics(self, metrics_summary: Dict):
        """Plot performance metrics dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Metric values
        metrics = {
            'False Positive Rate': metrics_summary['false_positive_rate'],
            'False Negative Rate': metrics_summary['false_negative_rate'],
            'Avg Detection Delay': metrics_summary['avg_detection_delay'],
            'Avg Processing Time': metrics_summary['avg_processing_time'],
            'AUPC': metrics_summary['area_under_performance_curve'],
            'Avg Accuracy Improvement': metrics_summary['avg_accuracy_improvement']
        }
        
        # Create gauge charts for each metric
        for idx, (metric_name, value) in enumerate(metrics.items()):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Create a simple gauge visualization
            self._create_gauge(ax, metric_name, value)
        
        plt.suptitle('Performance Metrics Dashboard', fontsize=18, fontweight='bold')
        return fig
    
    def _create_gauge(self, ax, title, value):
        """Create a simple gauge visualization"""
        ax.clear()
        
        # Determine color based on metric type and value
        if 'Rate' in title and value < 0.1:
            color = '#4ECDC4'  # Good (low error rate)
        elif 'Rate' in title and value > 0.3:
            color = '#FF6B6B'  # Bad (high error rate)
        elif 'Improvement' in title and value > 0:
            color = '#4ECDC4'  # Good (positive improvement)
        else:
            color = '#45B7D1'  # Neutral
        
        # Create a simple bar chart as gauge
        ax.barh([0], [value], color=color, alpha=0.8, height=0.5)
        ax.set_xlim(0, 1 if 'Rate' in title else max(1, value * 1.2))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.text(value/2, 0, f'{value:.3f}', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white')
        
        ax.grid(True, alpha=0.3, axis='x')