import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict

class AspectFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.kmeans = KMeans(n_clusters=config.N_ASPECT_CLUSTERS)
        
    def fit_transform(self, df: pd.DataFrame, aspects: List[str], word2vec_model) -> pd.DataFrame:
        self.fit(aspects, word2vec_model)
        return self.transform(df, aspects, word2vec_model)
    
    def fit(self, aspects: List[str], word2vec_model):
        aspect_vectors = [self.get_word_vector(word, word2vec_model) for word in aspects]
        self.kmeans.fit(aspect_vectors)
        
    def transform(self, df: pd.DataFrame, aspects: List[str], word2vec_model) -> pd.DataFrame:
        features = {}
        
        # Calculate all features
        features['NAW'] = self._calculate_naw(df, aspects)
        features['PAC'] = self._calculate_pac(df, aspects)
        features['ARL'] = self._calculate_arl(df, aspects)
        features['ARD'] = self._calculate_ard(df)
        features['ASD'] = self._calculate_asd(df)
        features['TNAW'] = self._calculate_tnaw(df, aspects)
        features['APAC'] = self._calculate_apac(df, aspects)
        features['AARL'] = self._calculate_aarl(df, aspects)
        
        return pd.DataFrame(features)
    
    def get_word_vector(self, word: str, model) -> np.ndarray:
        try:
            return model[word]
        except:
            return np.zeros(model.vector_size)
            
    # Feature calculation methods...
    def _calculate_naw(self, df: pd.DataFrame, aspects: List[str]) -> pd.Series:
        return df['reviewText'].apply(lambda x: len(set(x.split()) & set(aspects)))
    
    # Add other feature calculation methods...