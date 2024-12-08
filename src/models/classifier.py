import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict

class SpamClassifier:
    def __init__(self, config):
        self.config = config
        self.model = xgb.XGBClassifier(**config.XGB_PARAMS)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
            
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        predictions = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }