from data.data_loader import load_amazon_reviews, preprocess_reviews, ReviewDataset
from models.aspect_extractor import AspectExtractor 
from models.feature_extractor import AspectFeatureExtractor
from models.classifier import SpamClassifier
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

def main():
    # Load and preprocess data
    df = load_amazon_reviews('Cell_Phones_and_Accessories.json')
    df = preprocess_reviews(df)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    aspect_extractor = AspectExtractor()
    feature_extractor = AspectFeatureExtractor()
    classifier = SpamClassifier()
    
    # Extract aspects
    train_aspects = aspect_extractor.extract_aspects(train_df['reviewText'], tokenizer)
    
    # Extract features
    X_train = feature_extractor.extract_features(train_df, train_aspects)
    X_test = feature_extractor.extract_features(test_df, train_aspects)
    
    # Train and evaluate
    classifier.train(X_train, train_df['class'])
    results = classifier.evaluate(X_test, test_df['class'])
    print(f"Test Results: {results}")

if __name__ == "__main__":
    main()