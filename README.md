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
