import numpy as np
import matplotlib.pyplot as plt

# TODO 1: Read decay_observed.csv into arrays t and observed
t, observed = np.loadtxt('decay_observed.csv', delimiter=',', skiprows=1, unpack=True)

# TODO 2: Set N0 to the first observed value and compute analytical curve
LAMBDA = 0.3
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# TODO 3: Create 1x2 subplot with shared x and y axes
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))

# Left subplot: Observed data (points)
ax1.scatter(t, observed, color='blue', label='Observed')
ax1.set_title('Observed Data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Count')
ax1.grid(True)

# Right subplot: Analytical law (line)
ax2.plot(t, analytical, color='red', label='Analytical')
ax2.set_title('Analytical Law ($N_0 e^{-\lambda t}$)')
ax2.set_xlabel('Time')
ax2.grid(True)

plt.tight_layout()

# TODO 4: Save the figure as figure.png
plt.savefig('figure.png')
print("Saved figure.png successfully.")
