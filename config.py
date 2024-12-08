from pathlib import Path

class Config:
    # Paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "src/data"
    MODEL_DIR = ROOT_DIR / "src/models"
    
    # Data
    MAX_SEQUENCE_LENGTH = 128
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model parameters
    HIDDEN_SIZE = 768
    N_ASPECT_CLUSTERS = 12
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    
    # XGBoost parameters
    XGB_PARAMS = {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 10,
        'min_child_weight': 1,
        'subsample': 0.95,
        'colsample_bytree': 0.95,
        'reg_alpha': 0.01,
        'gamma': 0.4
    }