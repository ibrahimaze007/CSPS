import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance

os.makedirs("results", exist_ok=True)

# A2. Clean and look
df_raw = pd.read_csv("data/concrete.csv")
dup_count = df_raw.duplicated().sum()
na_count = df_raw.isna().sum().sum()
print(f"=== A2. Cleaning ===\nDuplicates: {dup_count}, Missing values: {na_count}")

df = df_raw.drop_duplicates().reset_index(drop=True)
X = df.drop(columns=["strength"])
y = df["strength"]
print("\n=== Summary Statistics ===")
print(df.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(df["cement"], df["strength"], alpha=0.4, color="steelblue")
axes[0].set_xlabel("Cement (kg/m³)")
axes[0].set_ylabel("Compressive Strength (MPa)")
axes[0].set_title("Strength vs. Cement")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df["age"], df["strength"], alpha=0.4, color="darkorange")
axes[1].set_xlabel("Age (days)")
axes[1].set_ylabel("Compressive Strength (MPa)")
axes[1].set_title("Strength vs. Age")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/A2_relationships.png")
plt.close()

# A3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n=== A3. Train/Test Split ===\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# A4. Linear Regression
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
y_tr_pred = lr.predict(X_train)
y_te_pred = lr.predict(X_test)

def reg_metrics(y_tr, p_tr, y_te, p_te, n_tr, p_feat):
    tr_rmse = np.sqrt(mean_squared_error(y_tr, p_tr))
    te_rmse = np.sqrt(mean_squared_error(y_te, p_te))
    te_mae = mean_absolute_error(y_te, p_te)
    tr_r2 = r2_score(y_tr, p_tr)
    tr_adj_r2 = 1 - (1 - tr_r2) * (n_tr - 1) / (n_tr - p_feat - 1)
    te_r2 = r2_score(y_te, p_te)
    return tr_rmse, te_rmse, te_mae, tr_r2, tr_adj_r2, te_r2

m_lin = reg_metrics(y_train, y_tr_pred, y_test, y_te_pred, len(X_train), X_train.shape[1])
print("\n=== A4. Linear Regression ===")
print(f"Intercept: {lr.intercept_:.4f}")
for col, c in zip(X.columns, lr.coef_):
    print(f"  {col:13s}: {c:+.5f}")
print(f"Train RMSE={m_lin[0]:.3f}, Test RMSE={m_lin[1]:.3f}, Test MAE={m_lin[2]:.3f}, Train R2={m_lin[3]:.4f}, Adj R2={m_lin[4]:.4f}, Test R2={m_lin[5]:.4f}")

residuals = y_test - y_te_pred
plt.figure(figsize=(6.5, 4))
plt.scatter(X_test["age"], residuals, alpha=0.6, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xscale("log")
plt.xlabel("Age (days, log scale)")
plt.ylabel("Residuals (True - Predicted MPa)")
plt.title("Linear Regression Test Residuals vs. Age")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/A4_residuals.png")
plt.close()

# A5. Polynomial Regression
print("\n=== A5. Polynomial Regression ===")
for deg in [1, 2, 3, 4]:
    poly = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), StandardScaler(), LinearRegression())
    poly.fit(X_train, y_train)
    p_cnt = poly[0].n_output_features_
    pm = reg_metrics(y_train, poly.predict(X_train), y_test, poly.predict(X_test), len(X_train), p_cnt)
    print(f"Deg {deg} (p={p_cnt:3d}): Train RMSE={pm[0]:.3f}, Test RMSE={pm[1]:.3f}, Train R2={pm[3]:.4f}, Adj R2={pm[4]:.4f}, Test R2={pm[5]:.4f}")

# A6. Classification: KNN & Decision Tree
y_class = (df["strength"] >= 40).astype(int)
Xc_train, Xc_test, c_train, c_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

print("\n=== A6. KNN Sweep ===")
Ks = [1, 3, 5, 7, 15, 25, 51, 101]
tr_accs, te_accs = [], []
for k in Ks:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(Xc_train, c_train)
    tr_accs.append(knn.score(Xc_train, c_train))
    te_accs.append(knn.score(Xc_test, c_test))
    print(f"K={k:3d}: Train Acc={tr_accs[-1]:.4f}, Test Acc={te_accs[-1]:.4f}")

plt.figure(figsize=(6.5, 4))
plt.plot(Ks, tr_accs, "o-", label="Train Accuracy")
plt.plot(Ks, te_accs, "s-", label="Test Accuracy")
plt.xscale("log")
plt.xlabel("K (log scale)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs. K")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/A6_knn_sweep.png")
plt.close()

chosen_K = 5
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_K)).fit(Xc_train, c_train)
knn_unscaled = KNeighborsClassifier(n_neighbors=chosen_K).fit(Xc_train, c_train)
print(f"At K={chosen_K}: Scaled Test Acc = {knn_scaled.score(Xc_test, c_test):.4f}, Unscaled Test Acc = {knn_unscaled.score(Xc_test, c_test):.4f}")

seed_accs = []
for rs in [0, 1, 2, 3, 4]:
    Xtr_s, Xte_s, ctr_s, cte_s = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
    m_s = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_K)).fit(Xtr_s, ctr_s)
    seed_accs.append(m_s.score(Xte_s, cte_s))
print(f"5-seed Test Accuracies (K={chosen_K}): {[round(a, 4) for a in seed_accs]} | Mean={np.mean(seed_accs):.4f}, Spread={np.max(seed_accs)-np.min(seed_accs):.4f}")

print("\n=== A6. Decision Tree Sweep ===")
depths = [1, 2, 3, 4, 5, 6, 8, None]
dt_tr, dt_te = [], []
for d in depths:
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0).fit(Xc_train, c_train)
    dt_tr.append(dt.score(Xc_train, c_train))
    dt_te.append(dt.score(Xc_test, c_test))
    print(f"max_depth={str(d):4s} (leaves={dt.get_n_leaves():3d}): Train Acc={dt_tr[-1]:.4f}, Test Acc={dt_te[-1]:.4f}")

plt.figure(figsize=(6.5, 4))
x_labels = [str(d) for d in depths]
plt.plot(x_labels, dt_tr, "o-", label="Train Accuracy")
plt.plot(x_labels, dt_te, "s-", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs. Depth")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/A6_tree_sweep.png")
plt.close()

chosen_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=1, random_state=0).fit(Xc_train, c_train)
print("\nTop 2 levels of chosen tree (max_depth=5):")
print(export_text(chosen_tree, feature_names=list(X.columns), max_depth=2))

# A7. Metrics & Thresholds
print("=== A7. Classification Metrics ===")
rows = []
for name, model in [("KNN (K=5)", knn_scaled), ("Decision Tree (depth=5)", chosen_tree)]:
    preds = model.predict(Xc_test)
    tn, fp, fn, tp = confusion_matrix(c_test, preds).ravel()
    rows.append({
        "Model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": round(accuracy_score(c_test, preds), 4),
        "Precision": round(precision_score(c_test, preds), 4),
        "Recall": round(recall_score(c_test, preds), 4),
        "F1": round(f1_score(c_test, preds), 4)
    })
metrics_df = pd.DataFrame(rows)
print(metrics_df.to_string(index=False))
metrics_df.to_csv("results/A_metrics.csv", index=False)

print("\nThreshold sweep on Decision Tree (depth=5):")
probs = chosen_tree.predict_proba(Xc_test)[:, 1]
for th in [0.3, 0.5, 0.7, 0.9]:
    p_th = (probs >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, p_th).ravel()
    print(f"th={th:.1f}: TP={tp:2d}, FP={fp:2d}, FN={fn:2d}, TN={tn:3d} | Prec={precision_score(c_test, p_th, zero_division=0):.4f}, Rec={recall_score(c_test, p_th):.4f}")

# A8. Feature Importance
std_lr = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
perm_knn = permutation_importance(knn_scaled, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
perm_dt = permutation_importance(chosen_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Raw_Coef": np.round(lr.coef_, 4),
    "Std_Coef": np.round(std_lr[-1].coef_, 4),
    "Tree_Imp": np.round(chosen_tree.feature_importances_, 4),
    "Perm_KNN": np.round(perm_knn.importances_mean, 4),
    "Perm_Tree": np.round(perm_dt.importances_mean, 4)
})
print("\n=== A8. Feature Importance ===")
print(imp_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].barh(imp_df["Feature"], np.abs(imp_df["Std_Coef"]), color="steelblue")
axes[0].set_title("|Standardised Linear Coef|")
axes[1].barh(imp_df["Feature"], imp_df["Tree_Imp"], color="forestgreen")
axes[1].set_title("Decision Tree Importance")
axes[2].barh(imp_df["Feature"], imp_df["Perm_Tree"], color="darkorange")
axes[2].set_title("Permutation Importance (Tree F1)")
plt.tight_layout()
plt.savefig("results/A8_importances.png")
plt.close()

# Bonus: Physics-guided features
X_phys1 = X.copy()
X_phys1["log_age"] = np.log(X_phys1["age"])
X_phys1 = X_phys1.drop(columns=["age"])
Xtr_p1, Xte_p1, ytr_p1, yte_p1 = train_test_split(X_phys1, y, test_size=0.2, random_state=42)
r2_p1 = r2_score(yte_p1, LinearRegression().fit(Xtr_p1, ytr_p1).predict(Xte_p1))

X_phys2 = X_phys1.copy()
binder = X_phys2["cement"] + X_phys2["slag"] + X_phys2["ash"]
X_phys2["w_b"] = X_phys2["water"] / binder
X_phys2 = X_phys2.drop(columns=["water"])
Xtr_p2, Xte_p2, ytr_p2, yte_p2 = train_test_split(X_phys2, y, test_size=0.2, random_state=42)
r2_p2 = r2_score(yte_p2, LinearRegression().fit(Xtr_p2, ytr_p2).predict(Xte_p2))
print(f"\n=== Bonus: Physics Features ===\nWith log(age) Test R2 = {r2_p1:.4f} (p=8); With log(age) + w/b Test R2 = {r2_p2:.4f} (p=8)")
