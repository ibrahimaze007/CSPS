import numpy as np
from scipy.optimize import newton, minimize

# --- Part 2A: f(x) = (x - 3)^2 + 1 ---
def f(x):
    return (x - 3)**2 + 1

def df(x):
    return 2 * (x - 3)

def d2f(x):
    return 2.0

# 1. Gradient descent by hand
x_gd = 0.0
lr = 0.1
for _ in range(200):
    x_gd -= lr * df(x_gd)

# 2. Newton's method on f'(x) = 0
x_newton = newton(df, x0=0.0, fprime=d2f)

# 3. SLSQP
res_slsqp = minimize(f, x0=0.0, method="SLSQP")

print("=== Part 2A: f(x) = (x - 3)^2 + 1 ===")
print(f"Gradient Descent: x = {x_gd:.6f}")
print(f"Newton's Method:  x = {x_newton:.6f}")
print(f"SLSQP:            x = {res_slsqp.x[0]:.6f}\n")


# --- Part 2B: g(x) = x^4 - 3x^2 + x + 5 ---
def g(x):
    return x**4 - 3 * x**2 + x + 5

def dg(x):
    return 4 * x**3 - 6 * x + 1

def d2g(x):
    return 12 * x**2 - 6

print("=== Part 2B: g(x) = x^4 - 3x^2 + x + 5 ===")
for x0 in [0.0, 2.0]:
    # 1. Gradient descent
    xg = x0
    for _ in range(500):
        xg -= 0.02 * dg(xg)

    # 2. Newton's method
    xn = newton(dg, x0=x0, fprime=d2g)
    curv = d2g(xn)
    kind = "minimum" if curv > 0 else "maximum"

    # 3. SLSQP
    rs = minimize(g, x0=x0, method="SLSQP")

    print(f"Starting from x0 = {x0}:")
    print(f"  Gradient Descent: x = {xg:.6f}, g(x) = {g(xg):.6f}")
    print(f"  Newton's Method:  x = {xn:.6f}, g''(x) = {curv:.4f} ({kind})")
    print(f"  SLSQP:            x = {rs.x[0]:.6f}, g(x) = {g(rs.x[0]):.6f}")
