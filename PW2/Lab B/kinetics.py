import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

t, C_meas = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1, unpack=True)
C0 = C_meas[0]

def total_error(k):
    k_val = k[0] if isinstance(k, (list, np.ndarray)) else k
    C_pred = C0 * np.exp(-k_val * t)
    return np.sum((C_meas - C_pred)**2)

res = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
k_fit = res.x[0]
print(f"Fitted rate constant k = {k_fit:.4f}")

t_smooth = np.linspace(t.min(), t.max(), 200)
plt.figure(figsize=(7, 4.5))
plt.scatter(t, C_meas, label="Measured Concentration", color="tab:blue")
plt.plot(t_smooth, C0 * np.exp(-k_fit * t_smooth), label=f"Fitted Curve (k = {k_fit:.4f})", color="tab:red")
plt.xlabel("Time (t)")
plt.ylabel("Concentration C(t)")
plt.title("First-Order Reaction Kinetics Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kinetics.png")
