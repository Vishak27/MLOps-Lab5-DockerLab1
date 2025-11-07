#Docker Lab1 - Wine Classification with Logistic Regression

import os
import json
import numpy as np
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load Dataset
wine = load_wine()
X = wine.data
y = wine.target
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class names: {wine.target_names}")
print()

# Data Splitting
# Read environment variables with defaults
test_size = float(os.getenv('TEST_SIZE', '0.2'))
random_state = int(os.getenv('RANDOM_STATE', '42'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print()

# Data Preprocessing
print("Preprocessing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preprocessing completed (StandardScaler)")
print()

# Model Training
# Read hyperparameters from environment variables
max_iter = int(os.getenv('MAX_ITER', '1000'))
solver = os.getenv('SOLVER', 'lbfgs')
model_random_state = int(os.getenv('RANDOM_STATE', '42'))

print(f"Training Logistic Regression with max_iter={max_iter}, solver={solver}...")
model = LogisticRegression(
    max_iter=max_iter,
    solver=solver,
    random_state=model_random_state
)
model.fit(X_train_scaled, y_train)
print("Model training completed")
print()

#Model Evaluation
print("=== Model Evaluation ===")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
print()

# Save Artifacts
# Create output directory structure
output_dir = 'outputs'
models_dir = os.path.join(output_dir, 'models')
results_dir = os.path.join(output_dir, 'results')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Save model
model_path = os.path.join(models_dir, 'wine_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(models_dir, 'wine_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved: {scaler_path}")

# Create results dictionary
confusion_matrix_list = confusion_matrix(y_test, y_pred).tolist()
results = {
    'accuracy': float(accuracy),
    'max_iter': max_iter,
    'solver': solver,
    'test_size': test_size,
    'random_state': random_state,
    'confusion_matrix': confusion_matrix_list,
    'class_names': wine.target_names.tolist()
}

# Save evaluation results to JSON
evaluation_path = os.path.join(results_dir, 'evaluation_results.json')
with open(evaluation_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Evaluation results saved: {evaluation_path}")
print()

print("All artifacts saved successfully!")