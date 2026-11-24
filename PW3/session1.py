import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load heart.csv
df = pd.read_csv("heart.csv")
print("=== Dataset 1: heart.csv ===")
print("Shape:", df.shape)
print(df.head(), "\n")

cols = ["age", "chol", "trestbps", "thalach"]
print(df[cols].describe(), "\n")

# 2. Visualise distributions (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flat, cols):
    ax.hist(df[col].dropna(), bins=20, edgecolor="black", alpha=0.75, color="steelblue")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("distributions.png")
print("Saved distributions.png")

# 3. Normality checks (Shapiro-Wilk + Skewness)
print("\n=== Normality Tests (Shapiro-Wilk) ===")
for col in cols:
    data = df[col].dropna()
    stat, p = stats.shapiro(data)
    skew = stats.skew(data)
    verdict = "Approximately Normal (p > 0.05)" if p > 0.05 else "Non-Normal (p <= 0.05)"
    print(f"{col:10s}: W = {stat:.4f}, p = {p:.4e}, skew = {skew:+.3f} -> {verdict}")
