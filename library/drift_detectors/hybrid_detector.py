from typing import List, Tuple, Dict
from pyspark.sql import DataFrame
from .kl_divergence import KLDivergenceDetector
from .hellinger_distance import HellingerDistanceDetector

class HybridDetector:
    """Hybrid drift detector combining KL Divergence and Hellinger Distance"""
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        self.alpha = alpha  # Weight for KL divergence
        self.beta = beta    # Weight for Hellinger distance
        self.kl_detector = KLDivergenceDetector()
        self.hellinger_detector = HellingerDistanceDetector()
        self.kl_history = []
        self.hellinger_history = []
        self.hybrid_history = []
    
    def normalize_score(self, score: float, history: List[float]) -> float:
        """Normalize score using min-max scaling from history"""
        if len(history) < 2:
            return score
        
        min_val = min(history + [score])
        max_val = max(history + [score])
        
        if max_val - min_val == 0:
            return 0.5
        
        return (score - min_val) / (max_val - min_val)
    
    def calculate_hybrid_score(self, kl_score: float, hellinger_score: float) -> float:
        """Calculate weighted hybrid score"""
        # Normalize scores
        kl_normalized = self.normalize_score(kl_score, self.kl_history)
        hell_normalized = self.normalize_score(hellinger_score, self.hellinger_history)
        
        # Weighted combination
        hybrid_score = self.alpha * kl_normalized + self.beta * hell_normalized
        
        return hybrid_score
    
    def detect(self, reference_df: DataFrame, current_df: DataFrame) -> Tuple[float, float, float, Dict]:
        """Detect drift using hybrid approach"""
        # Get KL divergence score
        kl_score, kl_features = self.kl_detector.detect(reference_df, current_df)
        
        # Get Hellinger distance score
        hellinger_score, hell_features = self.hellinger_detector.detect(reference_df, current_df)
        
        # Calculate hybrid score
        hybrid_score = self.calculate_hybrid_score(kl_score, hellinger_score)
        
        # Update history
        self.kl_history.append(kl_score)
        self.hellinger_history.append(hellinger_score)
        self.hybrid_history.append(hybrid_score)
        
        # Keep history size manageable
        if len(self.kl_history) > 100:
            self.kl_history.pop(0)
            self.hellinger_history.pop(0)
            self.hybrid_history.pop(0)
        
        # Combine feature scores
        feature_scores = {
            'kl': kl_features,
            'hellinger': hell_features
        }
        
        return kl_score, hellinger_score, hybrid_score, feature_scores
    
    def get_adaptive_threshold(self, factor: float = 1.5) -> float:
        """Calculate adaptive threshold based on history"""
        if len(self.hybrid_history) < 5:
            return 0.1
        
        recent_scores = self.hybrid_history[-20:]
        mean = sum(recent_scores) / len(recent_scores)
        variance = sum((x - mean) ** 2 for x in recent_scores) / len(recent_scores)
        std_dev = variance ** 0.5
        
        threshold = mean + factor * std_dev
        return max(0.05, min(threshold, 0.5))