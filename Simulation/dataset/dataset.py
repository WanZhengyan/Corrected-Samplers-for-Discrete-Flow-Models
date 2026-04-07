import torch
# import gym
# import d4rl
import numpy as np


# ==================== NumPy (CPU) versions ====================

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


def ar1_next_vectorized(x_prev_col, rng):
    """Vectorized version of ar1_next for a full column of n samples."""
    n = x_prev_col.shape[0]
    result = np.zeros(n, dtype=int)
    
    # Mask for values in [3, 6]
    in_range = (x_prev_col >= 3) & (x_prev_col <= 6)
    out_range = ~in_range
    
    # Out-of-range: uniform random in [1, 8]
    n_out = out_range.sum()
    if n_out > 0:
        result[out_range] = rng.randint(1, 9, size=n_out)
    
    # In-range
    n_in = in_range.sum()
    if n_in > 0:
        u = rng.uniform(0, 1, size=n_in)
        is_local = u < 0.9
        n_local = is_local.sum()
        n_far = n_in - n_local
        
        in_range_indices = np.where(in_range)[0]
        local_indices = in_range_indices[is_local]
        far_indices = in_range_indices[~is_local]
        
        # Local: offset in [-2, 2], clipped to [1, 8]
        if n_local > 0:
            offsets = rng.randint(-2, 3, size=n_local)
            result[local_indices] = np.clip(x_prev_col[local_indices] + offsets, 1, 8)
        
        # Far: choose from values with |v - x_prev| > 2
        if n_far > 0:
            for idx in far_indices:
                other_values = [v for v in range(1, 9) if abs(v - x_prev_col[idx]) > 2]
                result[idx] = rng.choice(other_values)
    
    return result


def generate_3d_block(n, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    data = np.zeros((n, 3), dtype=int)
    data[:, 0] = rng.randint(1, 9, size=n)
    for dim in range(1, 3):
        data[:, dim] = ar1_next_vectorized(data[:, dim-1], rng)
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


def generate_3k_discrete_data_batch(n, K, num_batches, seed=None):
    """Pre-generate multiple batches of data at once and return as a single numpy array.
    Returns shape: (num_batches, n, 3*K)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    total_n = n * num_batches
    all_data = []
    for k in range(K):
        block = generate_3d_block(total_n, rng)
        all_data.append(block)
    data = np.concatenate(all_data, axis=1) - 1  # shape: (total_n, 3*K)
    return data.reshape(num_batches, n, 3 * K)


# ==================== PyTorch GPU versions ====================

def _ar1_next_gpu(x_prev_col, device):
    """
    Vectorized AR(1) transition on GPU using PyTorch.
    x_prev_col: (n,) long tensor with values in [1, 8]
    Returns: (n,) long tensor with values in [1, 8]
    """
    n = x_prev_col.shape[0]
    result = torch.zeros(n, dtype=torch.long, device=device)

    in_range = (x_prev_col >= 3) & (x_prev_col <= 6)
    out_range = ~in_range

    # Out-of-range: uniform in [1, 8]
    n_out = out_range.sum().item()
    if n_out > 0:
        result[out_range] = torch.randint(1, 9, (n_out,), device=device)

    # In-range
    n_in = in_range.sum().item()
    if n_in > 0:
        u = torch.rand(n_in, device=device)
        is_local = u < 0.9

        in_range_indices = torch.where(in_range)[0]
        local_indices = in_range_indices[is_local]
        far_indices = in_range_indices[~is_local]

        # Local: offset in [-2, 2], clipped to [1, 8]
        n_local = local_indices.shape[0]
        if n_local > 0:
            offsets = torch.randint(-2, 3, (n_local,), device=device)
            result[local_indices] = (x_prev_col[local_indices] + offsets).clamp(1, 8)

        # Far: for each value in [3,6], precompute the "far" candidates
        # val -> candidates where |v - val| > 2, v in [1..8]
        n_far = far_indices.shape[0]
        if n_far > 0:
            # Precomputed far candidates for values 3,4,5,6
            # val=3: |v-3|>2 -> v in {1}: wait, |1-3|=2 not >2. v in {6,7,8}
            # val=4: v in {1,7,8}
            # val=5: v in {1,2,8}
            # val=6: v in {1,2,3}
            far_candidates = {
                3: torch.tensor([6, 7, 8], device=device),
                4: torch.tensor([1, 7, 8], device=device),
                5: torch.tensor([1, 2, 8], device=device),
                6: torch.tensor([1, 2, 3], device=device),
            }
            far_vals = x_prev_col[far_indices]
            for val, candidates in far_candidates.items():
                mask_val = (far_vals == val)
                count = mask_val.sum().item()
                if count > 0:
                    chosen = candidates[torch.randint(len(candidates), (count,), device=device)]
                    result[far_indices[mask_val]] = chosen

    return result


def _generate_3d_block_gpu(n, device):
    """Generate one 3-column AR(1) block directly on GPU. Values in [1, 8]."""
    data = torch.zeros(n, 3, dtype=torch.long, device=device)
    data[:, 0] = torch.randint(1, 9, (n,), device=device)
    for dim in range(1, 3):
        data[:, dim] = _ar1_next_gpu(data[:, dim - 1], device)
    return data


def generate_3k_discrete_data_gpu(n, K, device):
    """
    Generate data directly on GPU as a Long tensor.
    Returns: (n, 3*K) long tensor with values in [0, 7] (i.e. shifted by -1).
    """
    blocks = []
    for _ in range(K):
        block = _generate_3d_block_gpu(n, device)
        blocks.append(block)
    data = torch.cat(blocks, dim=1) - 1  # shift to [0, 7]
    return data

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, np_data):
        self.data = torch.from_numpy(np_data).long()
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return self.data.shape[0]