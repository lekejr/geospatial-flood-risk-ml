"""
Step 18: Create two key visualizations:
1. A bar chart comparing Logistic Regression vs Random Forest across
   all metrics.
2. A feature importance plot from the Random Forest model.
Also regenerates and saves confusion matrix plots for both models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# Reload test data and models
test_df = pd.read_csv("data/processed/test_set.csv")
train_df = pd.read_csv("data/processed/train_set.csv")
FEATURES = ["elevation", "slope", "rainfall", "ndvi", "ndwi", "landcover"]
X_test = test_df[FEATURES]
y_test = test_df["flooded"]

log_model = joblib.load("outputs/models/logistic_regression.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")
rf_model = joblib.load("outputs/models/random_forest.pkl")

X_test_scaled = scaler.transform(X_test)

# Predictions
log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# --- Plot 1: Metric comparison bar chart ---
metrics = {
    "Accuracy": [accuracy_score(y_test, log_pred), accuracy_score(y_test, rf_pred)],
    "Precision": [precision_score(y_test, log_pred), precision_score(y_test, rf_pred)],
    "Recall": [recall_score(y_test, log_pred), recall_score(y_test, rf_pred)],
    "F1-score": [f1_score(y_test, log_pred), f1_score(y_test, rf_pred)],
    "ROC-AUC": [roc_auc_score(y_test, log_prob), roc_auc_score(y_test, rf_prob)],
}

metric_names = list(metrics.keys())
log_scores = [metrics[m][0] for m in metric_names]
rf_scores = [metrics[m][1] for m in metric_names]

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, log_scores, width, label='Logistic Regression', color='#5b9bd5')
ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='#2e7d32')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/model_comparison.png", dpi=200)
print("Saved: outputs/figures/model_comparison.png")
plt.show()

# --- Plot 2: Feature importance (Random Forest) ---
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='#2e7d32')
ax.set_xlabel('Importance')
ax.set_title('Random Forest — Feature Importance')
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance.png", dpi=200)
print("Saved: outputs/figures/feature_importance.png")
plt.show()

# --- Plot 3: Confusion matrices side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_log = confusion_matrix(y_test, log_pred)
disp_log = ConfusionMatrixDisplay(cm_log, display_labels=["Not Flooded", "Flooded"])
disp_log.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Logistic Regression")

cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=["Not Flooded", "Flooded"])
disp_rf.plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrices.png", dpi=200)
print("Saved: outputs/figures/confusion_matrices.png")
plt.show()