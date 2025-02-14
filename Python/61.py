
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
def load_data(file_path):
    print("Loading financial transactions dataset...")
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    print("Preprocessing data...")
    df = df.dropna()
    
    if "Class" in df.columns:
        X = df.drop(columns=["Class"])
        y = df["Class"]
    else:
        raise ValueError("Dataset must have a 'Class' column for fraud detection.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance using SMOTE
    smote = SMOTE(sampling_strategy=0.3)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train XGBoost Model
def train_xgboost(X_train, y_train):
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, objective="binary:logistic")
    model.fit(X_train, y_train)
    return model

# Train Autoencoder for Anomaly Detection
def train_autoencoder(X_train):
    print("Training autoencoder for anomaly detection...")
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    dataset_path = "financial_transactions.csv"

    df = load_data(dataset_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train XGBoost model
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)

    # Train Autoencoder for anomaly detection
    autoencoder = train_autoencoder(X_train)
    autoencoder.save("fraud_autoencoder.h5")
    print("Fraud detection models saved successfully.")
