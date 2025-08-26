import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame, target_col: str):
    """Preprocess dataset: split features/target, scale, train-test split"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler