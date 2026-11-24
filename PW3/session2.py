import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")

# 4. Is the effect real? (thalach by target: 1 = disease, 0 = healthy)
healthy = df[df["target"] == 0]["thalach"].dropna()
disease = df[df["target"] == 1]["thalach"].dropna()

u_stat, p_mwu = stats.mannwhitneyu(disease, healthy, alternative="two-sided")
t_stat, p_ttest = stats.ttest_ind(disease, healthy, equal_var=False)

mean_h, sem_h = healthy.mean(), stats.sem(healthy)
mean_d, sem_d = disease.mean(), stats.sem(disease)

print("=== Task 4: Max Heart Rate (thalach) by Heart Disease ===")
print(f"Healthy (target=0): mean = {mean_h:.2f} ± {sem_h:.2f} (SEM), n = {len(healthy)}")
print(f"Disease (target=1): mean = {mean_d:.2f} ± {sem_d:.2f} (SEM), n = {len(disease)}")
print(f"Mann-Whitney U = {u_stat:.1f}, p = {p_mwu:.4e}")
print(f"Welch t-test   = {t_stat:.4f}, p = {p_ttest:.4e}\n")

plt.figure(figsize=(6, 4.5))
groups = ["Healthy (0)", "Heart Disease (1)"]
means = [mean_h, mean_d]
errors = [1.96 * sem_h, 1.96 * sem_d]
plt.bar(groups, means, yerr=errors, capsize=8, color=["#4c72b0", "#dd8452"], alpha=0.85, width=0.5)
plt.ylabel("Max Heart Rate (thalach)")
plt.title("Mean thalach with 95% Confidence Interval")
plt.ylim(0, max(means) + 30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("thalach_comparison.png")

# 5. Do age and thalach move together?
r_pearson, p_pearson = stats.pearsonr(df["age"], df["thalach"])
r_spearman, p_spearman = stats.spearmanr(df["age"], df["thalach"])
print("=== Task 5: Age vs thalach ===")
print(f"Pearson  r = {r_pearson:.4f} (p = {p_pearson:.4e})")
print(f"Spearman r = {r_spearman:.4f} (p = {p_spearman:.4e})\n")

plt.figure(figsize=(6.5, 4.5))
plt.scatter(df["age"], df["thalach"], alpha=0.6, color="teal")
m, b = np.polyfit(df["age"], df["thalach"], 1)
plt.plot(df["age"], m * df["age"] + b, color="red", linestyle="--", label=f"Linear fit (r = {r_pearson:.2f})")
plt.xlabel("Age (years)")
plt.ylabel("Max Heart Rate (thalach)")
plt.title("Age vs. Maximum Heart Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("age_vs_thalach.png")

# 6 & 7. Dataset 2: chemicals_cancer.csv (Confounding variable)
chem = pd.read_csv("chemicals_cancer.csv")
print("=== Task 6: Naive Analysis (All Patients) ===")
for c in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(chem[c], chem["malignancy"])
    print(f"  {c:8s} vs malignancy: r = {r:+.4f} (p = {p:.4e})")

print("\n=== Task 7: Controlling for Confounders ===")
for conf in ["pollution_index", "age"]:
    if conf in chem.columns:
        q40, q60 = chem[conf].quantile([0.4, 0.6])
        sub = chem[(chem[conf] >= q40) & (chem[conf] <= q60)]
        print(f"Restricted to middle {conf} [{q40:.1f}, {q60:.1f}] (n={len(sub)}):")
        for c in ["benzene", "cadmium"]:
            r, p = stats.pearsonr(sub[c], sub["malignancy"])
            print(f"  {c:8s} vs malignancy: r = {r:+.4f} (p = {p:.4e})")

# Bonus: Shannon entropy of target column
probs = df["target"].value_counts(normalize=True)
entropy_bits = -np.sum(probs * np.log2(probs))
print("\n=== Bonus: Target Category Balance ===")
print(probs.to_string())
print(f"Shannon Entropy = {entropy_bits:.4f} bits (max = 1.0000 bit)")
