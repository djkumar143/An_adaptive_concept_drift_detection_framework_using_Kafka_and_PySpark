from pyspark.sql import DataFrame
from pyspark.storagelevel import StorageLevel
from typing import Optional, Tuple

class AdaptiveWindow:
    """Adaptive sliding window management for drift detection"""
    
    def __init__(self, initial_size: int = 1000, min_size: int = 200, max_size: int = 2000):
        self.reference_size = initial_size
        self.current_size = initial_size // 2
        self.min_size = min_size
        self.max_size = max_size
        self.reference_window = None
        self.current_window = None
        self.adjustment_factor = 0.8  # Reduction factor when drift detected
        self.growth_factor = 1.1      # Growth factor when stable
    
    def update_windows(self, new_batch: DataFrame) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Update sliding windows with new batch"""
        if new_batch.count() == 0:
            return self.reference_window, self.current_window
        
        # Persist new batch for efficiency
        new_batch = new_batch.persist(StorageLevel.MEMORY_AND_DISK)
        
        # Initialize reference window if empty
        if self.reference_window is None:
            self.reference_window = new_batch
            return self.reference_window, None
        
        # Update current window
        if self.current_window is None:
            self.current_window = new_batch
        else:
            # Limit current window size before union
            current_count = self.current_window.count()
            if current_count >= self.current_size:
                # Keep only recent data
                rows_to_keep = max(self.current_size // 2, 1)
                self.current_window = self.current_window.limit(rows_to_keep)
            
            # Union with new batch
            self.current_window = self.current_window.union(new_batch)
            self.current_window = self.current_window.persist(StorageLevel.MEMORY_AND_DISK)
        
        # Manage window sizes
        self._manage_window_sizes()
        
        return self.reference_window, self.current_window
    
    def _manage_window_sizes(self):
        """Ensure windows don't exceed maximum sizes"""
        if self.reference_window and self.reference_window.count() > self.reference_size:
            self.reference_window = self.reference_window.limit(self.reference_size)
            self.reference_window = self.reference_window.persist(StorageLevel.MEMORY_AND_DISK)
        
        if self.current_window and self.current_window.count() > self.current_size:
            self.current_window = self.current_window.limit(self.current_size)
            self.current_window = self.current_window.persist(StorageLevel.MEMORY_AND_DISK)
    
    def adjust_size(self, drift_detected: bool):
        """Adjust window size based on drift detection"""
        if drift_detected:
            # Decrease size for better sensitivity
            self.current_size = max(self.min_size, int(self.current_size * self.adjustment_factor))
        else:
            # Increase size for efficiency
            self.current_size = min(self.max_size, int(self.current_size * self.growth_factor))
    
    def reset_current_window(self):
        """Reset current window after drift detection"""
        if self.current_window:
            self.current_window.unpersist()
        
        # Make current window the new reference
        self.reference_window = self.current_window
        self.current_window = None
    
    def get_window_sizes(self) -> Tuple[int, int]:
        """Get current window sizes"""
        ref_size = self.reference_window.count() if self.reference_window else 0
        curr_size = self.current_window.count() if self.current_window else 0
        return ref_size, curr_size