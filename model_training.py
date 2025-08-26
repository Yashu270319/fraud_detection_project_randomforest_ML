from sklearn.ensemble import RandomForestClassifier
import joblib

def build_model():
    """Initialize Random Forest model with chosen hyperparameters"""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced_subsample"
    )

def train_model(model, X_train, y_train):
    """Train the model"""
    model.fit(X_train, y_train)
    return model

def save_model(model, path: str):
    """Save trained model"""
    joblib.dump(model, path)

def load_model(path: str):
    """Load model from file"""
    return joblib.load(path)