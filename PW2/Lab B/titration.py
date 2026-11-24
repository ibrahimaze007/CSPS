import numpy as np
import matplotlib.pyplot as plt

V, pH = np.loadtxt("titration.csv", delimiter=",", skiprows=1, unpack=True)
slope = np.gradient(pH, V)

eq_idx = np.argmax(slope)
V_eq = V[eq_idx]
print(f"Equivalence point volume: {V_eq:.2f} mL (max slope = {slope[eq_idx]:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

ax1.plot(V, pH, color="tab:blue")
ax1.axvline(V_eq, color="red", linestyle="--", label=f"Eq. point = {V_eq:.1f} mL")
ax1.set_title("Titration Curve")
ax1.set_xlabel("Volume of Base (mL)")
ax1.set_ylabel("pH")
ax1.legend()
ax1.grid(True)

ax2.plot(V, slope, color="tab:orange")
ax2.axvline(V_eq, color="red", linestyle="--", label=f"Peak at {V_eq:.1f} mL")
ax2.set_title("First Derivative (dpH/dV)")
ax2.set_xlabel("Volume of Base (mL)")
ax2.set_ylabel("Slope (dpH/dV)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("titration.png")
