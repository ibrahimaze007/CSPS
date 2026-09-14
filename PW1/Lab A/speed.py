import time
import decay

atoms = 200000
rate = 0.4

start = time.perf_counter()
res_loop = decay.simulate(atoms, rate)
t_loop = time.perf_counter() - start

start = time.perf_counter()
res_numpy = decay.simulate_numpy(atoms, rate)
t_numpy = time.perf_counter() - start

speedup = t_loop / t_numpy if t_numpy > 0 else 1.0

print(f'Loop time: {t_loop:.6f} s')
print(f'NumPy time: {t_numpy:.6f} s')
print(f'Speed-up: {speedup:.2f}x faster')
