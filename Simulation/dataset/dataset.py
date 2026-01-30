import torch
# import gym
# import d4rl
import numpy as np


def ar1_next(x_prev, rng):
    if 3 <= x_prev <= 6:
        u = rng.uniform(0, 1)
        if u < 0.9:
            offset = rng.randint(-2, 3)
            return np.clip(x_prev + offset, 1, 8)
        else:
            other_values = [v for v in range(1, 9) if abs(v - x_prev) > 2]
            return rng.choice(other_values)
    else:
        return rng.randint(1, 9)

def generate_3d_block(n, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    data = np.zeros((n, 3), dtype=int)
    data[:, 0] = rng.randint(1, 9, size=n)
    for dim in range(1, 3):
        for i in range(n):
            data[i, dim] = ar1_next(data[i, dim-1], rng)
    return data

def generate_3k_discrete_data(n, K, seed=None):

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    all_data = []
    
    for k in range(K):
        block = generate_3d_block(n, rng)
        all_data.append(block)
    data = np.concatenate(all_data, axis=1) - 1
    
    return data

