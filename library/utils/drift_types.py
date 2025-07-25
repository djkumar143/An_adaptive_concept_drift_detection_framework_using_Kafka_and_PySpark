import numpy as np
from typing import List, Optional
from scipy import stats

class DriftTypeIdentifier:
    """Identify the type of concept drift based on score patterns"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.score_window = []
        self.thresholds = {
            'sudden': {'magnitude': 0.3, 'slope': 0.1},
            'gradual': {'magnitude': 0.2, 'slope': 0.05, 'duration': 5},
            'incremental': {'magnitude': 0.15, 'slope': 0.01, 'variability': 0.05}
        }
    
    def update(self, score: float):
        """Update the score window with new score"""
        self.score_window.append(score)
        if len(self.score_window) > self.window_size:
            self.score_window.pop(0)
    
    def identify_drift_type(self) -> Optional[str]:
        """Identify drift type based on score patterns"""
        if len(self.score_window) < 5:
            return None
        
        # Calculate statistical features
        scores = np.array(self.score_window)
        x = np.arange(len(scores))
        
        # Linear regression for trend
        slope, intercept, r_value, _, _ = stats.linregress(x, scores)
        
        # Calculate variability
        variability = np.std(scores)
        
        # Recent score magnitude
        recent_score = scores[-1]
        recent_avg = np.mean(scores[-3:])
        
        # Sudden drift: High magnitude with sharp increase
        if (recent_score > self.thresholds['sudden']['magnitude'] and 
            slope > self.thresholds['sudden']['slope']):
            return 'sudden'
        
        # Gradual drift: Moderate magnitude with consistent increase
        elif (self.thresholds['gradual']['magnitude'] < recent_avg <= self.thresholds['sudden']['magnitude'] and
              slope > self.thresholds['gradual']['slope'] and
              len(self.score_window) >= self.thresholds['gradual']['duration']):
            return 'gradual'
        
        # Incremental drift: Low magnitude with small consistent changes
        elif (recent_avg <= self.thresholds['incremental']['magnitude'] and
              abs(slope) < self.thresholds['incremental']['slope'] and
              variability < self.thresholds['incremental']['variability']):
            return 'incremental'
        
        return 'unknown'
    
    def get_drift_characteristics(self) -> dict:
        """Get detailed characteristics of the detected drift"""
        if len(self.score_window) < 2:
            return {}
        
        scores = np.array(self.score_window)
        x = np.arange(len(scores))
        
        # Calculate various statistics
        slope, intercept, r_value, _, _ = stats.linregress(x, scores)
        
        characteristics = {
            'current_score': scores[-1],
            'average_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'trend_slope': slope,
            'trend_strength': abs(r_value),
            'variability': np.std(scores),
            'rate_of_change': np.mean(np.diff(scores)) if len(scores) > 1 else 0
        }
        
        return characteristics
    
    def reset(self):
        """Reset the drift type identifier"""
        self.score_window = []