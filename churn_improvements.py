# %% [markdown]
# # Churn model: better evaluation
# Fixes three weaknesses of the first version:
# 1. Tuning used accuracy, which favours the majority class -> tune for F1 on churners.
# 2. One small test set (121 churners) -> add 5-fold stratified cross-validation.
# 3. The decision threshold is chosen on training data only, then the test set is used once.
# Also adds permutation feature importance for the final model.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_fscore_support)
from sklearn.inspection import permutation_importance

DATA_PATH = "bigml_59c28831336c6604c800002a.csv"   # change if your path differs
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load and prepare (same steps as the original notebook)
# The four `total_*_charge` columns are dropped: each is just minutes x a fixed rate,
# so keeping both would split their importance between two copies of the same signal.

# %%
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
df = df.drop(columns=["phone_number",
                      "total_day_charge", "total_eve_charge",
                      "total_night_charge", "total_intl_charge"])
for col in ["international_plan", "voice_mail_plan"]:
    df[col] = (df[col] == "yes").astype(int)
df["churn"] = df["churn"].astype(int)
df = pd.get_dummies(df, columns=["state"], prefix="state", dtype=int)

X = df.drop(columns="churn")
y = df["churn"]
# Same split as the original notebook (same seed, same stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=RANDOM_STATE, stratify=y)   # default test_size = 0.25

print(f"Train {X_train.shape}, test {X_test.shape}; churn rate {y.mean():.3f}")
print(f"Majority-class baseline accuracy: {1 - y_test.mean():.3f}")

# %% [markdown]
# ## 2. Tune for F1 on the churn class (not accuracy), with stratified 5-fold CV
# No feature scaling: tree models do not need it.

# %%
try:
    from xgboost import XGBClassifier
    MODEL_NAME = "XGBoost"
    base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    param_grid = {"learning_rate": [0.01, 0.1, 0.5],
                  "max_depth": [3, 5, 7],
                  "n_estimators": [100, 200, 300]}
except ImportError:                                  # fallback so the script still runs
    from sklearn.ensemble import HistGradientBoostingClassifier
    MODEL_NAME = "HistGradientBoosting (stand-in; install xgboost for the real run)"
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    param_grid = {"learning_rate": [0.01, 0.1, 0.5],
                  "max_depth": [3, 5, 7],
                  "max_iter": [100, 200, 300]}
print("Model:", MODEL_NAME)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(base, param_grid, scoring="f1", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print(f"Cross-validated F1 (churn class): {grid.best_score_:.3f}")

# %% [markdown]
# ## 3. Choose the decision threshold WITHOUT looking at the test set
# Out-of-fold probabilities on the training data stand in for unseen customers.

# %%
oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
rows = []
for t in np.arange(0.2, 0.75, 0.05):
    p, r, f, _ = precision_recall_fscore_support(
        y_train, (oof_proba >= t).astype(int), average="binary", zero_division=0)
    rows.append({"threshold": round(t, 2), "precision": p, "recall": r, "f1": f})
thr_table = pd.DataFrame(rows)
print(thr_table.round(3).to_string(index=False))
best_threshold = float(thr_table.loc[thr_table["f1"].idxmax(), "threshold"])
print(f"\nThreshold chosen on training data (max F1): {best_threshold}")

# %% [markdown]
# ## 4. Final evaluation on the held-out test set (used once)

# %%
test_proba = model.predict_proba(X_test)[:, 1]
for name, t in [("Default threshold 0.50", 0.5), (f"Chosen threshold {best_threshold}", best_threshold)]:
    pred = (test_proba >= t).astype(int)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["stayed", "churned"]))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.3f}")
print(f"PR-AUC (average precision): {average_precision_score(y_test, test_proba):.3f}"
      f"   (a random model scores about {y_test.mean():.3f})")

# %% [markdown]
# ## 5. What drives the predictions? Permutation importance on the test set
# Measures how much F1 drops when one feature's values are shuffled.

# %%
perm = permutation_importance(model, X_test, y_test, scoring="f1",
                              n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
imp = (pd.DataFrame({"feature": X_test.columns,
                     "importance": perm.importances_mean,
                     "std": perm.importances_std})
       .sort_values("importance", ascending=False))
print(imp.head(10).round(4).to_string(index=False))

top = imp.head(10).iloc[::-1]
plt.figure(figsize=(8, 5))
plt.barh(top["feature"], top["importance"], xerr=top["std"], color="#2a6f97")
plt.xlabel("Drop in F1 when the feature is shuffled")
plt.title("Top 10 features for predicting churn")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")
