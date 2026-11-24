import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 16.0  # Gives x = 2/3 ≈ 0.6667 and HI ≈ 1.33 mol

def k_imbalance(x):
    return (2 * x)**2 / ((1 - x) * (1 - x)) - K

# 1. Root-finding with Newton's method
x_newton = newton(k_imbalance, x0=0.5)

# 2. Minimization of k_imbalance(x)**2 with SLSQP
res = minimize(lambda x: k_imbalance(x[0])**2, x0=[0.5], method="SLSQP", bounds=[(0.01, 0.99)])
x_slsqp = res.x[0]

print(f"Equilibrium extent x (Newton): {x_newton:.6f}")
print(f"Equilibrium extent x (SLSQP):  {x_slsqp:.6f}")
print(f"Equilibrium amounts -> H2: {1 - x_newton:.4f} mol, I2: {1 - x_newton:.4f} mol, HI: {2 * x_newton:.4f} mol")

x_vals = np.linspace(0, 0.95, 200)
plt.figure(figsize=(7, 4.5))
plt.plot(x_vals, 1 - x_vals, label="H2 (1 - x)", linestyle="-")
plt.plot(x_vals, 1 - x_vals, label="I2 (1 - x)", linestyle="--")
plt.plot(x_vals, 2 * x_vals, label="HI (2x)", color="tab:green")
plt.axvline(x_newton, color="red", linestyle=":", label=f"Equilibrium x = {x_newton:.3f}")
plt.xlabel("Reaction Extent (x)")
plt.ylabel("Amount (mol)")
plt.title("Chemical Equilibrium: H2 + I2 <=> 2 HI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("equilibrium.png")
