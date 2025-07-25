import math
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count
from typing import Dict, Tuple

class KLDivergenceDetector:
    """KL Divergence based drift detector"""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.feature_names = ["at1", "at2", "at3", "cl"]
        self.num_bins = 10
    
    def calculate_probability_distribution(self, df: DataFrame, feature: str) -> Dict:
        """Calculate probability distribution for a feature"""
        total_count = df.count()
        if total_count == 0:
            return {}
        
        if feature in ["at1", "at2", "at3"]:
            # Numeric features - create bins
            stats = df.select(feature).summary("min", "max").collect()
            min_val = float(stats[0][feature])
            max_val = float(stats[1][feature])
            
            if min_val == max_val:
                return {0: 1.0}
            
            bin_width = (max_val - min_val) / self.num_bins
            
            # Create histogram
            df_binned = df.select(
                ((col(feature) - min_val) / bin_width).cast("int").alias("bin")
            )
            histogram = df_binned.groupBy("bin").count().collect()
            
            prob_dist = {
                row["bin"]: row["count"] / total_count 
                for row in histogram
            }
        else:
            # Categorical features
            value_counts = df.groupBy(feature).count().collect()
            prob_dist = {
                row[feature]: row["count"] / total_count 
                for row in value_counts
            }
        
        return prob_dist
    
    def calculate_kl_divergence(self, p_dist: Dict, q_dist: Dict) -> float:
        """Calculate KL divergence between two distributions"""
        if not p_dist or not q_dist:
            return 0.0
        
        kl_div = 0.0
        all_keys = set(p_dist.keys()).union(set(q_dist.keys()))
        
        for key in all_keys:
            p_val = p_dist.get(key, self.epsilon)
            q_val = q_dist.get(key, self.epsilon)
            
            if p_val > 0:
                kl_div += p_val * math.log(p_val / q_val)
        
        return kl_div
    
    def detect(self, reference_df: DataFrame, current_df: DataFrame) -> Tuple[float, Dict]:
        """Detect drift using KL divergence"""
        kl_scores = []
        feature_scores = {}
        
        for feature in self.feature_names:
            p_dist = self.calculate_probability_distribution(reference_df, feature)
            q_dist = self.calculate_probability_distribution(current_df, feature)
            
            kl_div = self.calculate_kl_divergence(p_dist, q_dist)
            kl_scores.append(kl_div)
            feature_scores[feature] = kl_div
        
        # Aggregate score
        overall_score = sum(kl_scores) / len(kl_scores) if kl_scores else 0
        
        return overall_score, feature_scores