from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from typing import Optional, Tuple
import time

class HoeffdingTreeModel:
    """Hoeffding Tree implementation using Decision Tree as proxy"""
    
    def __init__(self, max_depth: int = 5, min_instances: int = 10):
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.model = None
        self.warning_model = None
        self.assembler = VectorAssembler(
            inputCols=["at1", "at2", "at3"], 
            outputCol="features"
        )
        self.evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
    
    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """Preprocess data for model training"""
        # Convert class to numeric label
        df = df.withColumn("label", df["cl"].cast("double"))
        
        # Create feature vector
        df = self.assembler.transform(df)
        return df.select("features", "label", "at1", "at2", "at3", "cl")
    
    def train(self, data: DataFrame) -> Tuple[float, float]:
        """Train the model and return accuracy and training time"""
        start_time = time.time()
        
        # Use Decision Tree as Hoeffding Tree alternative
        classifier = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=self.max_depth,
            minInstancesPerNode=self.min_instances,
            impurity="gini"
        )
        
        self.model = classifier.fit(data)
        training_time = time.time() - start_time
        
        # Evaluate accuracy
        predictions = self.model.transform(data)
        accuracy = self.evaluator.evaluate(predictions)
        
        return accuracy, training_time
    
    def train_warning_model(self, data: DataFrame):
        """Train a warning model in parallel"""
        classifier = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=self.max_depth,
            minInstancesPerNode=self.min_instances,
            impurity="gini"
        )
        self.warning_model = classifier.fit(data)
    
    def evaluate(self, data: DataFrame) -> float:
        """Evaluate model on data"""
        if self.model is None or data.count() == 0:
            return 0.0
        
        predictions = self.model.transform(data)
        return self.evaluator.evaluate(predictions)
    
    def predict(self, data: DataFrame) -> DataFrame:
        """Make predictions on data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.transform(data)
    
    def switch_to_warning_model(self):
        """Switch to pre-trained warning model"""
        if self.warning_model is not None:
            self.model = self.warning_model
            self.warning_model = None
            return True
        return False
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the model"""
        if self.model is None:
            return {}
        
        # For Decision Tree, we can get feature importances
        importances = self.model.featureImportances.toArray()
        return {
            "at1": float(importances[0]),
            "at2": float(importances[1]),
            "at3": float(importances[2])
        }