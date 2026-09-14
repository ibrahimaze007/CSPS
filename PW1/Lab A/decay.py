import numpy as np

def simulate(atoms=1000, rate=0.4):
    if rate < 0:
        raise ValueError('Rate cannot be negative')
    current = atoms
    history = [current]
    while current > 0:
        decayed = sum(1 for _ in range(current) if np.random.rand() < rate)
        current -= decayed
        history.append(current)
        if current == 0 or len(history) > 100:
            break
    return history

def simulate_numpy(atoms=1000, rate=0.4):
    if rate < 0:
        raise ValueError('Rate cannot be negative')
    current = atoms
    history = [current]
    while current > 0:
        draws = np.random.random(current)
        decayed = np.count_nonzero(draws < rate)
        current -= decayed
        history.append(current)
        if current == 0 or len(history) > 100:
            break
    return history
