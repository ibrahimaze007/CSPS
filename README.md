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
