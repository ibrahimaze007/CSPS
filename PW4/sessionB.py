import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

os.makedirs("results", exist_ok=True)

# B1. Load and hide labels
wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target
print("=== B1. Wine Dataset Summary ===")
print("Shape:", X.shape)
print(X.describe().T[["min", "max", "mean", "std"]].round(3))

# B2. PCA on raw data
pca_raw = PCA().fit(X)
print(f"\n=== B2. Raw PCA ===\nPC1 explained variance ratio: {pca_raw.explained_variance_ratio_[0]:.4f}")
load_raw = pd.Series(pca_raw.components_[0], index=X.columns).reindex(
    pd.Series(pca_raw.components_[0], index=X.columns).abs().sort_values(ascending=False).index
)
print("Top PC1 raw loadings:\n", load_raw.head(5).round(4))

# B3. Scale, then PCA again
scaler = StandardScaler()
Z = scaler.fit_transform(X)
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
n_90 = np.argmax(cum_evr >= 0.90) + 1
print(f"\n=== B3. Scaled PCA ===\nComponents for >=90% variance: {n_90} (cum={cum_evr[n_90-1]:.4f})")
print(f"PC1+PC2 variance: {cum_evr[1]:.4f} (PC1={evr[0]:.4f}, PC2={evr[1]:.4f})")

load_df = pd.DataFrame({
    "PC1": np.round(pca.components_[0], 4),
    "PC2": np.round(pca.components_[1], 4)
}, index=X.columns)
print("\nPC1 and PC2 Loadings (sorted by |PC1|):")
print(load_df.reindex(load_df["PC1"].abs().sort_values(ascending=False).index))

fig, ax = plt.subplots(figsize=(7, 4))
comps = np.arange(1, len(evr) + 1)
ax.bar(comps, evr, alpha=0.7, label="Individual Variance")
ax.plot(comps, cum_evr, "ro-", label="Cumulative Variance")
ax.axhline(0.90, color="green", linestyle="--", label="90% Threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Scaled Wine PCA Scree Plot")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/B3_scree_plot.png")
plt.close()

scores = pca.transform(Z)
plt.figure(figsize=(6, 4.5))
plt.scatter(scores[:, 0], scores[:, 1], color="gray", alpha=0.75)
plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
plt.title("Wines in PC1-PC2 Plane (Unlabelled)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/B3_pca_unlabelled.png")
plt.close()

# B4. K-Means
print("\n=== B4. K-Means Stability & Elbow ===")
inertias_rs = [KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=rs).fit(Z).inertia_ for rs in range(8)]
print("Inertias (n_init=1, rs=0..7):", [round(x, 2) for x in inertias_rs])

ks = range(1, 11)
inertias_k = [KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(Z).inertia_ for k in ks]
plt.figure(figsize=(6.5, 4))
plt.plot(ks, inertias_k, "o-", color="navy")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve (Scaled Wine Data)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/B4_elbow.png")
plt.close()

# B5. Cluster centres
km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(Z)
print("\n=== B5. Cluster Sizes & Centres (k=3) ===")
print("Cluster sizes:", np.bincount(km.labels_))
centres_real = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns).round(2)
print(centres_real.T)

plt.figure(figsize=(6, 4.5))
plt.scatter(scores[:, 0], scores[:, 1], c=km.labels_, cmap="viridis", alpha=0.85)
plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
plt.title("Wines in PC1-PC2 Plane Coloured by K-Means (k=3)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/B5_pca_clusters.png")
plt.close()

# B6. The reveal
def agreement(labels):
    ct = pd.crosstab(cultivar, labels)
    return ct, ct.max(axis=0).sum() / len(cultivar)

ct_scaled, ag_scaled = agreement(km.labels_)
km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(X)
_, ag_raw = agreement(km_raw.labels_)
km_pca2 = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(scores[:, :2])
_, ag_pca2 = agreement(km_pca2.labels_)

print("\n=== B6. Crosstab (Scaled K-Means vs True Cultivar) ===")
print(ct_scaled)

ag_df = pd.DataFrame({
    "Method": ["K-Means on Scaled Data (13D)", "K-Means on Raw Unscaled Data (13D)", "K-Means on Scaled PC1-PC2 (2D)"],
    "Agreement": [round(ag_scaled, 4), round(ag_raw, 4), round(ag_pca2, 4)]
})
print("\n", ag_df.to_string(index=False))
ag_df.to_csv("results/B_agreement.csv", index=False)

# Bonus: Supervised vs Unsupervised
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)
sup_full = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xw_tr, yw_tr).score(Xw_te, yw_te)
sup_pca2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5)).fit(Xw_tr, yw_tr).score(Xw_te, yw_te)
print(f"\n=== Bonus: Supervised KNN Test Accuracy ===\nScaled 13D KNN(5): {sup_full:.4f} | Scaled PCA(2)+KNN(5): {sup_pca2:.4f}")
