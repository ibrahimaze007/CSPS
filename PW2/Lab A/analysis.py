import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Part 2: Read freefall.csv and compute velocity and acceleration
t, y = np.loadtxt("freefall.csv", delimiter=",", skiprows=1, unpack=True)

v = np.gradient(y, t)
a = np.gradient(v, t)

print(f"Mean acceleration: {a.mean():.4f} m/s^2")

# Part 3: Standard deviation of noisy acceleration
print(f"Acceleration std:  {a.std():.4f} m/s^2")

# Part 4: Integrate acceleration back to velocity, then position
v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

max_diff = np.max(np.abs(y - y_rec))
print(f"Max position difference: {max_diff:.4f} m")

# Part 5: 3 stacked panels sharing time axis (position, velocity, acceleration)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

ax1.plot(t, y, label="Measured position", color="tab:blue")
ax1.set_ylabel("Position y (m)")
ax1.set_title("Free Fall Motion: Position, Velocity, and Acceleration")
ax1.grid(True)

ax2.plot(t, v, label="Velocity (dy/dt)", color="tab:orange")
ax2.set_ylabel("Velocity v (m/s)")
ax2.grid(True)

ax3.plot(t, a, label="Acceleration (dv/dt)", color="tab:green")
ax3.axhline(-9.81, color="red", linestyle="--", label="Expected g = -9.81 m/s²")
ax3.set_ylabel("Acceleration a (m/s²)")
ax3.set_xlabel("Time t (s)")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("motion.png")
print("Saved motion.png successfully.")
