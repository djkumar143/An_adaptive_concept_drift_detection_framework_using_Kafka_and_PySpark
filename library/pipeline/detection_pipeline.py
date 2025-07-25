from pyspark.sql import DataFrame
import time
from typing import Tuple, Dict
from ..drift_detectors import HybridDetector
from ..window_strategy import AdaptiveWindow
from ..models import HoeffdingTreeModel
from ..metrics import PerformanceMetrics
from ..utils import DriftTypeIdentifier, DriftVisualizer

class DriftDetectionPipeline:
    """Main pipeline for drift detection and model adaptation"""
    
    def __init__(self, warning_threshold: float = 0.05, drift_threshold: float = 0.1):
        self.hybrid_detector = HybridDetector()
        self.window_manager = AdaptiveWindow()
        self.model = HoeffdingTreeModel()
        self.metrics = PerformanceMetrics()
        self.drift_identifier = DriftTypeIdentifier()
        self.visualizer = DriftVisualizer()
        
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.warning_detected = False
        
        # Score history for visualization
        self.kl_scores = []
        self.hellinger_scores = []
        self.hybrid_scores = []
        self.drift_points = []
        self.batch_count = 0
    
    def process_batch(self, batch_df: DataFrame, batch_id: int) -> Dict:
        """Process a single batch of data"""
        start_time = time.time()
        self.batch_count = batch_id
        
        if batch_df.count() == 0:
            return {"status": "empty_batch", "batch_id": batch_id}
        
        # Preprocess batch
        processed_batch = self.model.preprocess_data(batch_df)
        
        # Update windows
        ref_window, curr_window = self.window_manager.update_windows(processed_batch)
        
        if ref_window is None or curr_window is None:
            return {"status": "initializing", "batch_id": batch_id}
        
        # Get window sizes
        ref_size, curr_size = self.window_manager.get_window_sizes()
        
        # Detect drift
        kl_score, hell_score, hybrid_score, feature_scores = self.hybrid_detector.detect(
            ref_window, curr_window
        )
        
        # Update histories
        self.kl_scores.append(kl_score)
        self.hellinger_scores.append(hell_score)
        self.hybrid_scores.append(hybrid_score)
        
        # Update drift identifier
        self.drift_identifier.update(hybrid_score)
        
        # Get adaptive threshold
        adaptive_threshold = self.hybrid_detector.get_adaptive_threshold()
        
        # Check for drift
        drift_detected = hybrid_score > adaptive_threshold
        warning_detected = hybrid_score > self.warning_threshold and not drift_detected
        
        result = {
            "batch_id": batch_id,
            "kl_score": kl_score,
            "hellinger_score": hell_score,
            "hybrid_score": hybrid_score,
            "threshold": adaptive_threshold,
            "drift_detected": drift_detected,
            "warning_detected": warning_detected,
            "window_sizes": {"reference": ref_size, "current": curr_size},
            "feature_scores": feature_scores
        }
        
        # Handle drift detection
        if drift_detected:
            self.drift_points.append(batch_id)
            drift_type = self.drift_identifier.identify_drift_type()
            result["drift_type"] = drift_type
            
            # Record metrics
            acc_before = self.model.evaluate(curr_window) if self.model.model else 0
            
            # Adapt model
            if self.warning_detected and self.model.switch_to_warning_model():
                print(f"Switched to pre-trained warning model")
            else:
                acc, train_time = self.model.train(curr_window)
                result["training_time"] = train_time
            
            acc_after = self.model.evaluate(curr_window)
            
            # Record performance
            self.metrics.record_accuracy(acc_before, acc_after)
            self.metrics.record_drift_detection(drift_type or "unknown", hybrid_score)
            
            # Reset windows
            self.window_manager.reset_current_window()
            self.window_manager.adjust_size(True)
            
            # Reset warning state
            self.warning_detected = False
            
            result["accuracy_before"] = acc_before
            result["accuracy_after"] = acc_after
            
        elif warning_detected and not self.warning_detected:
            # Train warning model
            self.model.train_warning_model(curr_window)
            self.warning_detected = True
            result["warning_model_trained"] = True
        else:
            # No drift - adjust window size
            self.window_manager.adjust_size(False)
            if hybrid_score < self.warning_threshold * 0.8:
                self.warning_detected = False
        
        # Record processing time
        processing_time = time.time() - start_time
        self.metrics.record_processing_time(processing_time)
        result["processing_time"] = processing_time
        
        return result
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return self.metrics.get_summary()
    
    def get_visualization_data(self) -> Dict:
        """Get data for visualization"""
        return {
            "kl_scores": self.kl_scores,
            "hellinger_scores": self.hellinger_scores,
            "hybrid_scores": self.hybrid_scores,
            "drift_points": self.drift_points,
            "threshold": self.hybrid_detector.get_adaptive_threshold()
        }