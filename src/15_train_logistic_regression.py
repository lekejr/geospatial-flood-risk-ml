"""
Step 16: Train a Logistic Regression baseline model to predict flood
occurrence from elevation, slope, rainfall, NDVI, NDWI, and land cover.
Uses class_weight='balanced' to address the ~90/10 class imbalance.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

# Load train/test sets
train_df = pd.read_csv("data/processed/train_set.csv")
test_df = pd.read_csv("data/processed/test_set.csv")

# Define features and target
FEATURES = ["elevation", "slope", "rainfall", "ndvi", "ndwi", "landcover"]
TARGET = "flooded"

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# Scale features (important for Logistic Regression to converge properly
# and treat all features fairly, since they're on very different scales
# e.g., elevation in meters vs NDVI between -1 and 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with class balancing
print("Training Logistic Regression...")
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n=== Logistic Regression Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFull classification report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler for later use
joblib.dump(model, "outputs/models/logistic_regression.pkl")
joblib.dump(scaler, "outputs/models/scaler.pkl")
print("\nModel saved to outputs/models/logistic_regression.pkl")
print("Scaler saved to outputs/models/scaler.pkl")