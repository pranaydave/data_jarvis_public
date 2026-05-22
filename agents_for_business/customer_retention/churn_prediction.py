"""
Telco Customer Churn Prediction
Binary classification pipeline: Logistic Regression, Random Forest, Gradient Boosting
Includes SHAP explainability and model serialisation for the LangGraph agent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & inspect ────────────────────────────────────────────────────────
#Put your datapath
DATA_PATH = "./telco_customer_churn_train.csv"

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
print(f"\nChurn rate: {df['Churn'].mean():.1%}")

# ── 2. Clean ─────────────────────────────────────────────────────────────────

# TotalCharges can contain spaces for new customers (tenure=0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop non-informative ID column
df.drop(columns=["customerID"], inplace=True)

# ── 3. Feature engineering ───────────────────────────────────────────────────

# Bin tenure into customer lifecycle segments
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 mo", "12-24 mo", "24-48 mo", "48-72 mo"],
    include_lowest=True,
)

# Average monthly spend ratio
df["charges_ratio"] = df["TotalCharges"] / (df["tenure"].replace(0, 1))

# ── 4. Split features / target ───────────────────────────────────────────────

TARGET = "Churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify column types
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nNumeric features  : {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# ── 5. Preprocessor ──────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# ── 6. Models ────────────────────────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
}

# ── 7. Train / evaluate ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = {}

print("\n" + "=" * 60)
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()

    results[name] = {"pipe": pipe, "y_pred": y_pred, "y_prob": y_prob, "auc": auc, "cv_auc": cv_auc}

    print(f"\n── {name} ──")
    print(f"  Test AUC : {auc:.4f}")
    print(f"  CV AUC   : {cv_auc:.4f} (5-fold on train set)")
    print(classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"]))

# ── 8. Pick best model by test AUC ──────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\nBest model: {best_name}  (AUC {best['auc']:.4f})")

# ── 9. Plots ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Churn Prediction — Model Evaluation", fontsize=14)

# 9a. ROC curves
ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(fontsize=8)

# 9b. Confusion matrix for best model
ax = axes[1]
cm = confusion_matrix(y_test, best["y_pred"])
ConfusionMatrixDisplay(cm, display_labels=["Not Churn", "Churn"]).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix\n{best_name}")

# 9c. Feature importance (tree models) or coefficients (LR)
ax = axes[2]
best_pipe = best["pipe"]
clf = best_pipe.named_steps["clf"]
ohe_cols = best_pipe.named_steps["prep"].transformers_[1][1].get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numeric_cols, ohe_cols])

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    title = f"Feature Importances\n{best_name}"
elif hasattr(clf, "coef_"):
    importances = np.abs(clf.coef_[0])
    title = f"Feature Coefficients (|coef|)\n{best_name}"

top_idx = np.argsort(importances)[-15:]
ax.barh(feature_names[top_idx], importances[top_idx])
ax.set_title(title)
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("churn_evaluation.png", dpi=150)
print("\nPlot saved → churn_evaluation.png")
plt.show()

# ── 10. AUC comparison bar chart ─────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 4))
names = list(results.keys())
aucs = [results[n]["auc"] for n in names]
bars = ax2.bar(names, aucs, color=["#4c72b0", "#55a868", "#c44e52"])
ax2.set_ylim(0.5, 1.0)
ax2.set_ylabel("Test ROC-AUC")
ax2.set_title("Model Comparison")
for bar, auc in zip(bars, aucs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{auc:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("churn_model_comparison.png", dpi=150)
print("Plot saved → churn_model_comparison.png")
plt.show()

# ── 11. SHAP explainability ───────────────────────────────────────────────────

print("\nComputing SHAP values for best model…")

best_pipe = best["pipe"]
clf = best_pipe.named_steps["clf"]
prep = best_pipe.named_steps["prep"]

ohe_cols = prep.transformers_[1][1].get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numeric_cols, ohe_cols])

X_test_transformed = prep.transform(X_test)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_transformed)

# Beeswarm (summary) plot
fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test_transformed,
    feature_names=feature_names,
    plot_type="dot", show=False, max_display=20,
)
plt.title(f"SHAP Summary — {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
print("Plot saved → shap_summary.png")
plt.show()

# Bar plot — mean |SHAP|
fig_bar, ax_bar = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test_transformed,
    feature_names=feature_names,
    plot_type="bar", show=False, max_display=20,
)
plt.title(f"SHAP Mean |Value| — {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
print("Plot saved → shap_bar.png")
plt.show()

# ── 12. Save artefacts for the LangGraph agent ───────────────────────────────

joblib.dump(best_pipe, "churn_model.pkl")
joblib.dump(
    {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names.tolist(),
    },
    "churn_metadata.pkl",
)
print("\nModel saved → churn_model.pkl")
print("Metadata saved → churn_metadata.pkl")
