import numpy as np


class DelayClassifier:
    """
    A classifier wrapper that applies a custom threshold for binary delay prediction.
    
    Args:
        model: The underlying classifier model (e.g., CatBoost, LightGBM)
        threshold: Classification threshold for predicting class 1 (default: 0.6)
    """
    
    def __init__(self, model, threshold=0.6):
        self.model = model
        self.threshold = threshold

    def predict_proba(self, X):
        """
        Predict probability of class 1 (delay).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for class 1
        """
        return self.model.predict(X, prediction_type="Probability")[:, 1]

    def predict(self, X):
        """
        Predict binary class labels using the custom threshold.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)
