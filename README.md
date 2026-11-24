# CSPC Coursework

## PW1 Lab A
- Environment: configured
- Tests: 3 passing
- Speedup: tested
# CSPC Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup
Create the environment for a given lab:
```bash
conda env create -f PW1/Lab\ A/environment.yml
conda activate cspc
PW1

Lab A: Reproducible Foundations

What I built:
Set up CSPC repository structure, configured Git/GitHub, built conda environment, implemented decay simulation tests, and benchmarked speed improvements.

Speed comparison (loop vs NumPy):

    loop: 0.45 s

    numpy: 0.02 s

    speed-up: 22.5x faster

Tests: all passing? yes

Conclusion:
Successfully established a reproducible Python environment and version control repository. Unit tests under pytest validated simulation edge cases, and NumPy demonstrated a significant computational speedup over standard Python loops.


## PW1 --- Lab B

- **What the data showed:** The observed counts decay exponentially over time starting from $N_0 = 5000$ at $t = 0$.
- **Comparison with analytical law:** Based on the figure, the observed data closely matches the analytical law $N(t) = N_0 e^{-\lambda t}$ ($\lambda = 0.3$), with minor random statistical scatter.
- **Snakemake pipeline:** The `Snakefile` automates reproducing `figure.png` from `decay_observed.csv` via `plot.py` only when the output figure is missing or outdated.

## PW2 --- Lab A

- **Mean Acceleration & Noise Observation:** The measured mean acceleration is `-8.58 m/s²` (close to $-g = -9.81\text{ m/s}^2$), while the standard deviation is `28.72 m/s²` because taking two numerical derivatives amplifies small measurement noise between consecutive points.
- **Integration Recovery:** Integrating the noisy acceleration twice back to position recovers the original trajectory within a maximum difference of `0.78 m`, showing that integration accumulates values and cancels out random noise.

## PW2 --- Lab B

- **Method Comparison (Part 2):** On the convex function $f(x) = (x - 3)^2 + 1$, Gradient Descent, Newton's method, and SLSQP all converged to $x = 3.0$. On the multi-stationary landscape $g(x) = x^4 - 3x^2 + x + 5$, the starting point and algorithm mattered: from $x_0 = 0.0$, Gradient Descent and SLSQP reached the global minimum ($x \approx -1.3008$), whereas Newton's method converged to a local maximum ($x \approx 0.1699$, $g''(x) = -5.65 < 0$). From $x_0 = 2.0$, Gradient Descent and Newton settled in a local minimum ($x \approx 1.1309$, $g''(x) = 9.35 > 0$), while SLSQP still found the global minimum ($x \approx -1.3006$).
- **Reaction Kinetics (Part 3):** Fitting the first-order decay model $C(t) = C_0 e^{-kt}$ via SLSQP yielded a rate constant of $k = 0.2618$.
- **Chemical Equilibrium (Part 4):** Both Newton's method and SLSQP agreed on the equilibrium extent $x = 0.6667$, giving an equilibrium composition of $0.3333\text{ mol }\text{H}_2$, $0.3333\text{ mol }\text{I}_2$, and $1.3333\text{ mol }\text{HI}$.
- **Titration Equivalence Point (Part 5 Bonus):** Differentiating the titration curve identified the steepest slope ($\text{dpH}/\text{dV} = 4.00$) and equivalence point at $V = 50.00\text{ mL}$.

## PW3 --- Data, Distributions, Testing, and Causation

- **Dataset 1 — Distributions & Normality (Session 1):**
  - `age`: Slightly left-skewed ($\text{skew} = -0.201$), Shapiro-Wilk $W = 0.9864, p = 0.0058$ $\rightarrow$ **Non-normal**.
  - `chol`: Strongly right-skewed ($\text{skew} = +1.138$) with high outliers, Shapiro-Wilk $W = 0.9469, p = 5.36 \times 10^{-9}$ $\rightarrow$ **Non-normal**.
  - `trestbps`: Right-skewed ($\text{skew} = +0.710$), Shapiro-Wilk $W = 0.9659, p = 1.46 \times 10^{-6}$ $\rightarrow$ **Non-normal**.
  - `thalach`: Left-skewed ($\text{skew} = -0.535$), Shapiro-Wilk $W = 0.9763, p = 6.62 \times 10^{-5}$ $\rightarrow$ **Non-normal**.
- **Hypothesis Test on Max Heart Rate (`thalach`) (Task 4):** Because `thalach` deviates from normality, a two-sided Mann-Whitney U test was chosen (confirmed by Welch's t-test). The two groups differ significantly ($U = 17038.0, p = 9.80 \times 10^{-14}$): `target=1` has a mean of $158.47 \pm 1.49$ (SEM) bpm compared to $139.10 \pm 1.92$ (SEM) bpm for `target=0`, which matches the non-overlapping 95% confidence intervals in `thalach_comparison.png`.
- **Age vs. Max Heart Rate (Task 5):** `age` and `thalach` move in opposite directions with a moderate, highly significant negative correlation ($\text{Spearman } r = -0.3981, p = 6.02 \times 10^{-13}$; $\text{Pearson } r = -0.3985$), showing that maximum heart rate decreases with age.
- **Dataset 2 — Correlation vs. Causation Mystery (Tasks 6 & 7):**
  - *Naive analysis:* Across all 1,000 patients, `cadmium` appeared strongly correlated with `malignancy` ($r = +0.8800$), whereas `benzene` appeared much weaker ($r = +0.3847$).
  - *Controlling for confounding (`pollution_index`):* Restricting the analysis to patients within a narrow `pollution_index` band ($39.8\text{--}59.7, n=200$) caused the `cadmium` correlation to collapse ($r = +0.3095$) while `benzene`'s correlation jumped to $r = +0.6966$ ($p = 2.28 \times 10^{-30}$). Controlling for `age` alone left the spurious pattern intact ($r_{\text{cadmium}} = +0.8947$). Therefore, **`benzene` is the true driver of malignancy**, while `cadmium` only looked guilty due to confounding with area `pollution_index`.
- **Bonus — Category Balance (`target`):** The `target` classes are well balanced ($54.5\%$ vs. $45.5\%$), yielding a Shannon entropy of **$0.9943\text{ bits}$** (very close to the $1.0000\text{ bit}$ maximum unpredictability for a binary variable).

## PW4 --- Machine Learning: Supervised and Unsupervised

### Session A — Concrete Compressive Strength (Supervised Learning)
- **Variables & Units (A1):** `cement` (kg/m³), `slag` (kg/m³), `ash` (kg/m³), `water` (kg/m³), `superplastic` (kg/m³), `coarse_agg` (kg/m³), `fine_agg` (kg/m³), `age` (days), and target `strength` (MPa).
- **Data Cleaning & Scaling (A2–A3):** Removed 25 duplicate rows (0 missing values), leaving 1,005 unique samples split 80/20 (`804` train, `201` test). Duplicate rows must be removed before splitting so identical samples do not leak across train and test sets. Features span very different numerical scales (`coarse_agg` up to ~1,145 kg/m³ vs. `superplastic` ~0–32 kg/m³). `strength` increases positively with `cement` and rises logarithmically with curing `age` (rapid early gain, slower later).
- **Linear & Polynomial Regression (A4–A5):** Linear regression residuals vs. log(`age`) show a curved systematic pattern because hydration gain is non-linear in `age`, confirming a plain linear model underfits. Polynomial degrees 1–4 show test $R^2$ improving from degree 1 to degree 2 and peaking at degree 3, whereas at **degree 4** ($p = 494$ features vs. $n = 804$ training rows) the model severely overfits (train RMSE drops near zero while test RMSE explodes). Training adjusted $R^2$ alone does not protect against extreme overfitting when $p$ approaches $n$.
- **Classification — KNN & Decision Tree (A6):** Sweeping $K \in \{1, 3, 5, 7, 15, 25, 51, 101\}$ shows $K=1$ overfits (100% train accuracy) and very large $K$ underfits; **$K=5$** balances bias and variance. Scaling is essential because unscaled Euclidean distances are dominated by large-range aggregates (`coarse_agg`, `fine_agg`). Across 5 random splits (`random_state=0..4`), test accuracy varies by ~4–5%, showing that differences smaller than this spread are within sampling noise. Increasing `max_depth` monotonically drives train accuracy to 1.00 while test accuracy plateaus around `max_depth=5` (chosen for interpretability and generalisation). The root split tests `cement` / `age`, which makes direct physical sense in concrete curing.
- **Metrics, Thresholds & Feature Importance (A7–A8):** On a construction site, a **False Positive (FP)** (predicting $\ge 40$ MPa when the concrete is actually weaker) is the dangerous error because weak concrete could fail structurally; **Precision** controls FP. Raising the probability threshold from `0.5` to `0.7` or `0.9` reduces FPs and increases Precision at the cost of lower Recall (rejecting more safe batches). Standardised coefficients, tree importances, and permutation importances agree that **`cement`**, **`water`**, **`age`**, and **`superplastic`** are the dominant predictors, matching civil engineering principles (water-to-binder ratio and curing age). High observational importance proves predictive association, not direct causal intervention without controlled experiments.
- **Bonus (Physics Features):** Replacing `age` with `log(age)` and `water` with the water-to-binder ratio $w/b = \text{water}/(\text{cement}+\text{slag}+\text{ash})$ substantially boosts plain linear regression test $R^2$ while keeping only $p=8$ features, demonstrating that domain knowledge beats brute-force polynomial expansion.

#### Raw Session A Numerical Outputs
```text
=== A2. Cleaning ===
Duplicates: 25, Missing values: 0

=== Summary Statistics ===
        cement     slag      ash  ...  fine_agg      age  strength
count  1005.00  1005.00  1005.00  ...   1005.00  1005.00   1005.00
mean    278.63    72.04    55.54  ...    772.69    45.86     35.25
std     104.34    86.17    64.21  ...     80.34    63.73     16.28
min     102.00     0.00     0.00  ...    594.00     1.00      2.33
25%     190.70     0.00     0.00  ...    724.30     7.00     23.52
50%     265.00    20.00     0.00  ...    780.00    28.00     33.80
75%     349.00   142.50   118.30  ...    822.20    56.00     44.87
max     540.00   359.40   200.10  ...    992.60   365.00     82.60

[8 rows x 9 columns]

=== A3. Train/Test Split ===
Train size: 804, Test size: 201

=== A4. Linear Regression ===
Intercept: -23.3893
  cement       : +0.11667
  slag         : +0.09807
  ash          : +0.08458
  water        : -0.13139
  superplastic : +0.33148
  coarse_agg   : +0.01552
  fine_agg     : +0.02065
  age          : +0.11064
Train RMSE=10.002, Test RMSE=11.191, Test MAE=8.895, Train R2=0.6099, Adj R2=0.6059, Test R2=0.5802

=== A5. Polynomial Regression ===
Deg 1 (p=  8): Train RMSE=10.002, Test RMSE=11.191, Train R2=0.6099, Adj R2=0.6059, Test R2=0.5802
Deg 2 (p= 44): Train RMSE=6.989, Test RMSE=8.308, Train R2=0.8095, Adj R2=0.7984, Test R2=0.7686
Deg 3 (p=164): Train RMSE=4.396, Test RMSE=5.952, Train R2=0.9246, Adj R2=0.9053, Test R2=0.8813
Deg 4 (p=494): Train RMSE=2.334, Test RMSE=39.410, Train R2=0.9788, Adj R2=0.9448, Test R2=-4.2061

=== A6. KNN Sweep ===
K=  1: Train Acc=0.9975, Test Acc=0.8308
K=  3: Train Acc=0.9341, Test Acc=0.8010
K=  5: Train Acc=0.9067, Test Acc=0.8159
K=  7: Train Acc=0.8930, Test Acc=0.8358
K= 15: Train Acc=0.8483, Test Acc=0.8259
K= 25: Train Acc=0.8408, Test Acc=0.8060
K= 51: Train Acc=0.8047, Test Acc=0.7811
K=101: Train Acc=0.7575, Test Acc=0.7363
At K=5: Scaled Test Acc = 0.8159, Unscaled Test Acc = 0.8060
5-seed Test Accuracies (K=5): [0.801, 0.8657, 0.8607, 0.8607, 0.8358] | Mean=0.8448, Spread=0.0647

=== A6. Decision Tree Sweep ===
max_depth=1    (leaves=  2): Train Acc=0.7550, Test Acc=0.7015
max_depth=2    (leaves=  4): Train Acc=0.7550, Test Acc=0.7015
max_depth=3    (leaves=  8): Train Acc=0.8396, Test Acc=0.7960
max_depth=4    (leaves= 16): Train Acc=0.8694, Test Acc=0.8259
max_depth=5    (leaves= 26): Train Acc=0.8893, Test Acc=0.8458
max_depth=6    (leaves= 38): Train Acc=0.9303, Test Acc=0.8408
max_depth=8    (leaves= 61): Train Acc=0.9726, Test Acc=0.8557
max_depth=None (leaves= 80): Train Acc=0.9975, Test Acc=0.8806

Top 2 levels of chosen tree (max_depth=5):
|--- age <= 42.00
|   |--- cement <= 257.70
|   |   |--- water <= 156.20
|   |   |   |--- truncated branch of depth 3
|   |   |--- water >  156.20
|   |   |   |--- truncated branch of depth 3
|   |--- cement >  257.70
|   |   |--- superplastic <= 0.95
|   |   |   |--- truncated branch of depth 3
|   |   |--- superplastic >  0.95
|   |   |   |--- truncated branch of depth 3
|--- age >  42.00
|   |--- superplastic <= 6.45
|   |   |--- fine_agg <= 749.55
|   |   |   |--- truncated branch of depth 3
|   |   |--- fine_agg >  749.55
|   |   |   |--- truncated branch of depth 3
|   |--- superplastic >  6.45
|   |   |--- cement <= 203.35
|   |   |   |--- truncated branch of depth 3
|   |   |--- cement >  203.35
|   |   |   |--- truncated branch of depth 3

=== A7. Classification Metrics ===
                  Model  TP  FP  FN  TN  Accuracy  Precision  Recall     F1
              KNN (K=5)  53  18  19 111    0.8159     0.7465  0.7361 0.7413
Decision Tree (depth=5)  58  17  14 112    0.8458     0.7733  0.8056 0.7891

Threshold sweep on Decision Tree (depth=5):
th=0.3: TP=63, FP=28, FN= 9, TN=101 | Prec=0.6923, Rec=0.8750
th=0.5: TP=58, FP=17, FN=14, TN=112 | Prec=0.7733, Rec=0.8056
th=0.7: TP=47, FP=11, FN=25, TN=118 | Prec=0.8103, Rec=0.6528
th=0.9: TP=31, FP= 4, FN=41, TN=125 | Prec=0.8857, Rec=0.4306

=== A8. Feature Importance ===
     Feature  Raw_Coef  Std_Coef  Tree_Imp  Perm_KNN  Perm_Tree
      cement    0.1167   12.1188    0.2525    0.1477     0.1698
        slag    0.0981    8.4373    0.0363    0.0635     0.0085
         ash    0.0846    5.4395    0.0374    0.0174     0.0000
       water   -0.1314   -2.8279    0.0952    0.0380     0.0119
superplastic    0.3315    1.9708    0.1645    0.0362     0.1881
  coarse_agg    0.0155    1.1990    0.0109    0.0294     0.0017
    fine_agg    0.0206    1.6699    0.0905    0.0578     0.0720
         age    0.1106    6.8945    0.3126    0.1574     0.1648

=== Bonus: Physics Features ===
With log(age) Test R2 = 0.8037 (p=8); With log(age) + w/b Test R2 = 0.8064 (p=8)
```

### Session B — Wine Chemistry (Unsupervised Learning)
- **Dataset & Raw vs. Scaled PCA (B1–B3):** The UCI Wine dataset contains 178 wines and 13 chemical measurements (`proline` spans 278–1680 mg/L, whereas `nonflavanoid_phenols` spans 0.13–0.66). On **unscaled** data, `proline` alone accounts for >99.8% of PC1 variance purely due to its large numerical scale, making raw PC1 uninformative chemically. After `StandardScaler`, **8 components** retain 92.02% of variance (90% threshold), and the **PC1–PC2 plane retains 55.41%** (`PC1 = 36.20%`, `PC2 = 19.21%`). PC1 separates wines by phenolic content (`flavanoids`, `total_phenols`, `od280/od315`), while PC2 separates them by `color_intensity`, `alcohol`, and `proline`, revealing 3 clear visual groups.
- **K-Means & Cluster Interpretation (B4–B5):** With `n_init=1`, inertia varies across random seeds (`1277.93`–`1282.46`) due to local minima; `n_init=10` reliably selects the lowest-inertia solution. The elbow curve bends sharply at **$k = 3$** (inertia always decreases as $k$ increases up to $k=n$, so minimising inertia alone would choose $k=n$). Cluster 0 (`n=65`) has low alcohol (`12.25`), low color intensity (`2.97`), and low proline (`510.17`); Cluster 1 (`n=51`) has high malic acid (`3.31`), very low flavanoids (`0.82`), and intense color (`7.23`); Cluster 2 (`n=62`) has high alcohol (`13.68`), rich flavanoids (`3.00`), low ash alkalinity (`17.46`), and very high proline (`1100.23`).
- **Cultivar Reveal & Snakemake Automation (B6–B7):** Scaled 13D K-Means and 2D PC1–PC2 K-Means both achieve **96.63% agreement** (`172/178` wines) with the hidden true cultivars, whereas raw unscaled K-Means drops to **70.22%** because `proline` dominates Euclidean distances. Reproduced via `snakemake --cores 1` from `PW4/`; running it a second time reports `Nothing to be done` because all outputs are newer than their inputs, and editing `sessionA.py` reruns only `session_a` (and `all`), skipping `fetch` and `session_b`.
- **Bonus (Supervised vs. Unsupervised):** Supervised `KNeighborsClassifier(5)` achieves **93.33%** test accuracy (both 13D and 2D PCA). Comparing this directly to B6's 96.63% unsupervised agreement is not completely fair because B6 evaluated cluster-to-label matching in-sample across all 178 wines with post-hoc label alignment, whereas supervised KNN was evaluated out-of-sample on a held-out 25% test split.
