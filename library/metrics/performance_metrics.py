from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

class PerformanceMetrics:
    """Track and calculate performance metrics for drift detection"""
    
    def __init__(self):
        self.drift_detection_times = []
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.accuracy_before_drift = []
        self.accuracy_after_drift = []
        self.detection_delays = []
        self.processing_times = []
        self.drift_points = []
        self.detected_points = []
        
        # Drift type statistics
        self.drift_summary = {
            "sudden": {"detected": 0, "scores": [], "delays": []},
            "gradual": {"detected": 0, "scores": [], "delays": []},
            "incremental": {"detected": 0, "scores": [], "delays": []}
        }
    
    def record_drift_detection(self, drift_type: str, score: float, delay: int = 0):
        """Record a drift detection event"""
        if drift_type in self.drift_summary:
            self.drift_summary[drift_type]["detected"] += 1
            self.drift_summary[drift_type]["scores"].append(score)
            self.drift_summary[drift_type]["delays"].append(delay)
            self.detection_delays.append(delay)
    
    def record_accuracy(self, before: float, after: float):
        """Record accuracy before and after adaptation"""
        self.accuracy_before_drift.append(before)
        self.accuracy_after_drift.append(after)
    
    def record_processing_time(self, time: float):
        """Record processing time for a batch"""
        self.processing_times.append(time)
    
    def calculate_fpr_fnr(self) -> Tuple[float, float]:
        """Calculate false positive and false negative rates"""
        total_positives = self.true_positives + self.false_negatives
        total_negatives = self.true_negatives + self.false_positives
        
        fpr = self.false_positives / total_negatives if total_negatives > 0 else 0
        fnr = self.false_negatives / total_positives if total_positives > 0 else 0
        
        return fpr, fnr
    
    def calculate_aupc(self) -> float:
        """Calculate Area Under Performance Curve"""
        if not self.accuracy_before_drift or not self.accuracy_after_drift:
            return 0.0
        
        # Simple AUPC calculation
        improvements = [after - before for before, after in 
                       zip(self.accuracy_before_drift, self.accuracy_after_drift)]
        return np.mean(improvements) if improvements else 0.0
    
    def get_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        fpr, fnr = self.calculate_fpr_fnr()
        aupc = self.calculate_aupc()
        
        summary = {
            "total_drifts_detected": sum(self.drift_summary[dt]["detected"] for dt in self.drift_summary),
            "avg_detection_delay": np.mean(self.detection_delays) if self.detection_delays else 0,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "area_under_performance_curve": aupc,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "avg_accuracy_improvement": np.mean([a - b for a, b in zip(self.accuracy_after_drift, 
                                                                      self.accuracy_before_drift)]) 
                                       if self.accuracy_after_drift else 0,
            "drift_type_breakdown": {}
        }
        
        # Add drift type breakdown
        for drift_type, stats in self.drift_summary.items():
            if stats["detected"] > 0:
                summary["drift_type_breakdown"][drift_type] = {
                    "count": stats["detected"],
                    "avg_score": np.mean(stats["scores"]),
                    "avg_delay": np.mean(stats["delays"])
                }
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()