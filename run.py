import torch
from transformers import AutoTokenizer
from gensim.models import Word2Vec
from pathlib import Path

from config import Config
from src.data.data_loader import DataManager
from src.models.aspect_extractor import AspectExtractor
from src.models.feature_extractor import AspectFeatureExtractor
from src.models.classifier import SpamClassifier

def main():
    # Initialize config
    config = Config()
    
    # Initialize components
    data_manager = DataManager(config)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    aspect_extractor = AspectExtractor(config)
    feature_extractor = AspectFeatureExtractor(config)
    classifier = SpamClassifier(config)
    
    # Load and preprocess data
    data_path = config.DATA_DIR / "Cell_Phones_and_Accessories.json"
    df = data_manager.load_data(data_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_manager.create_data_loaders(df, tokenizer)
    
    # Train aspect extractor
    aspect_extractor.train(train_loader, val_loader)
    
    # Extract aspects
    train_aspects = aspect_extractor.extract_aspects(df['reviewText'], tokenizer)
    
    # Train word2vec model
    sentences = [text.split() for text in df['reviewText']]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
    
    # Extract features
    X_train = feature_extractor.fit_transform(df, train_aspects, word2vec_model.wv)
    
    # Train classifier
    classifier.train(
        X_train,
        df['class'],
        None,  # Add validation data if needed
        None
    )
    
    # Evaluate
    results = classifier.evaluate(X_train, df['class'])  # Use proper test set
    print(f"Results: {results}")

if __name__ == "__main__":
    main()