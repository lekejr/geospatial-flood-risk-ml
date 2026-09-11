"""
Step 17: Train a Random Forest model to predict flood occurrence,
using the same features and class-balancing strategy as the Logistic
Regression baseline, for a fair comparison.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

# Load train/test sets
train_df = pd.read_csv("data/processed/train_set.csv")
test_df = pd.read_csv("data/processed/test_set.csv")

FEATURES = ["elevation", "slope", "rainfall", "ndvi", "ndwi", "landcover"]
TARGET = "flooded"

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# Random Forest doesn't require feature scaling, unlike Logistic Regression
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1  # use all CPU cores
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFull classification report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# Save
joblib.dump(model, "outputs/models/random_forest.pkl")
print("\nModel saved to outputs/models/random_forest.pkl")