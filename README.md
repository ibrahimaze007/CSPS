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
