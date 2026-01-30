# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F

from flow_matching.path import MixtureDiscreteProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False




def get_nearest_times(time_grid: Tensor, t_discretization: Tensor) -> Tensor:
    """Get nearest times in t_discretization for each time in time_grid."""
    distances = torch.cdist(
        time_grid.unsqueeze(1),
        t_discretization.unsqueeze(1),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    nearest_indices = distances.argmin(dim=1)
    return t_discretization[nearest_indices]




class MixtureDiscreteUniformizationSolver(Solver):
    """Segmented uniformization solver for CTMC process defined by MixtureDiscreteProbPath."""
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        N: int = 100,
        delta: float = 1e-3,
        opt_grid: bool = True,
        mask: bool = False,
        dtype_categorical: torch.dtype = torch.float32,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Segmented uniformization sampling for CTMC.
        Args:
            x_init (Tensor): Initial state.
            t0 (float): Initial time.
            t1 (float): Final time.
            N (int): Number of segments in [0, 1-tau].
            tau (float): Early stopping threshold.
            dtype_categorical (torch.dtype): Precision for categorical sampler.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Extra model args.
        Returns:
            Tensor: Final state after segmented uniformization.
        
        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        device = x_init.device
        x_t = x_init.clone()
        dim = x_t.shape[1]
        batch_size, _ = x_t.shape

        if opt_grid:
            # Create grid: [1-delta^{(i-1)/N},1-delta^{i/N}], i=1,...,N
            t_grid = torch.tensor([1 - delta**(i/N) for i in range(0, N+1)], device=device)
        else:
            # Create uniform grid: [t0, t0+h, t0+2h, ..., t1-delta]
            # where h = (t1 - delta - t0) / N
            t_grid = torch.linspace(t0, t1 - delta, N + 1, device=device)

        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=N, desc=f"Uniformization")
        else:
            ctx = nullcontext()
        
        with ctx:
            for seg_idx in range(N):
                t_left = t_grid[seg_idx].item()
                t_right = t_grid[seg_idx+1].item()
                t_seg = torch.tensor([t_right], device=device)
                
                # Get rate at right endpoint
                scheduler_output = self.path.scheduler(t=t_seg)
                k_t = scheduler_output.alpha_t
                d_k_t = scheduler_output.d_alpha_t
                lambda_seg = torch.abs(d_k_t / (1 - k_t)).max().item()
                
                # Batch Poisson sampling: each sample gets its own n_jumps
                expected_jumps = dim * lambda_seg * (t_right - t_left)
                expected_jumps_tensor = torch.full((batch_size,), expected_jumps, device=device)
                n_jumps_batch = torch.distributions.poisson.Poisson(expected_jumps_tensor).sample().long()
                
                max_n_jumps = n_jumps_batch.max().item()
                if max_n_jumps == 0:
                    if verbose:
                        ctx.update(1)
                    continue
                
                # Batch uniform sampling: (batch_size, max_n_jumps)
                jump_times_batch = torch.rand(batch_size, max_n_jumps, device=device) * (t_right - t_left) + t_left
                
                # Create mask: (batch_size, max_n_jumps), True where jump is valid
                jump_mask = torch.arange(max_n_jumps, device=device).unsqueeze(0) < n_jumps_batch.unsqueeze(1)
                
                # Mask invalid jumps before sorting (set to inf so they sort to the end)
                jump_times_batch_masked = jump_times_batch.clone()
                jump_times_batch_masked[~jump_mask] = float('inf')
                
                # Sort: valid jump times will be sorted, invalid ones will be at the end
                jump_times_batch, _ = torch.sort(jump_times_batch_masked, dim=1)
                
                # Process all jumps in parallel
                for jump_idx in range(max_n_jumps):
                    # Get current jump time for all samples
                    t_jump_batch_raw = jump_times_batch[:, jump_idx]
                    current_mask = jump_mask[:, jump_idx]
                    
                    if not current_mask.any():
                        continue
                    
                    # Replace inf with valid value to avoid errors in model/scheduler
                    t_jump_batch = torch.where(
                        current_mask,
                        t_jump_batch_raw,
                        torch.full_like(t_jump_batch_raw, t_right)
                    )
                    
                    # Model prediction: each sample uses its own t_jump
                    p_1t = self.model(x=x_t, t=t_jump_batch, **model_extras)
                    
                    # Compute u_t(x,x_t): need to handle per-sample time
                    scheduler_output = self.path.scheduler(t=t_jump_batch)
                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t
                    
                    # Broadcast to (batch_size, dim, vocab_size)
                    k_t = k_t.view(batch_size, 1, 1)
                    d_k_t = d_k_t.view(batch_size, 1, 1)
                    u = d_k_t / (1 - k_t) * p_1t
                    
                    # Set u_t(x_t,x_t) = 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
                    
                    # Uniformization: sample new state with probability
                    probs = u / (lambda_seg * dim)
                    probs_flat = probs.view(batch_size, -1)
                    probs_sum = probs_flat.sum(dim=-1)
                    
                    # Decide whether to jump (only for valid samples)
                    jump_prob = torch.rand(batch_size, device=device) < probs_sum
                    mask_update = current_mask & jump_prob
                    
                    if mask_update.sum() > 0:
                        # Sample from flattened distribution
                        flat_indices = categorical(probs_flat[mask_update].to(dtype=dtype_categorical))
                        positions = flat_indices // self.vocabulary_size
                        values = flat_indices % self.vocabulary_size
                        batch_indices = torch.arange(batch_size, device=device)[mask_update]
                        x_t[batch_indices, positions] = values
                
                # Update progress after each segment
                if verbose:
                    ctx.update(1)
            if mask == True:
                mask_token = (x_t == self.vocabulary_size - 1)
                if mask_token.any() > 0:
                    t_jump_plus = torch.tensor([t_right], device=device)
                    # Get model prediction at final time
                    p_1t = self.model(x=x_t, t=t_jump_plus.repeat(batch_size), **model_extras)
                    
                    # Set probability of staying at current state to 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)
                    
                    # Sample new values from the distribution for mask positions
                    if mask_token.sum() > 0:
                        x_t[mask_token] = categorical(
                            p_1t[mask_token].to(dtype=dtype_categorical)
                        )
        
        return x_t, N


class MixtureDiscreteTauleapingSolver(Solver):
    """tau-leaping solver for CTMC process defined by MixtureDiscreteProbPath"""
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        N: int = 100,
        delta: float = 0.05,
        opt_grid: bool = True,
        mask: bool = False,
        dtype_categorical: torch.dtype = torch.float32,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Segmented tau-leaping sampling for CTMC.
        Args:
            x_init (Tensor): Initial state.
            t0 (float): Initial time.
            t1 (float): Final time.
            N (int): Number of segments in [0, 1-delta].
            delta (float): Early stopping threshold.
            opt_grid (bool): Whether to use optimized non-uniform grid. Defaults to True.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Extra model args.
        Returns:
            Tensor: Final state after segmented tau-leaping.
        
        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        device = x_init.device
        x_t = x_init.clone()
        batch_size, _ = x_t.shape
        
        if opt_grid:
            # Create grid: [1-delta^{(i-1)/N},1-delta^{i/N}], i=1,...,N
            t_grid = torch.tensor([1 - delta**(i/N) for i in range(0, N+1)], device=device)
        else:
            # Create uniform grid: [t0, t0+h, t0+2h, ..., t1-tau]
            # where h = (t1 - tau - t0) / N
            t_grid = torch.linspace(t0, t1 - delta, N + 1, device=device)

        
        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=N, desc=f"Tau-leaping")
        else:
            ctx = nullcontext()
        
        with ctx:
            for seg_idx in range(N):
                t_left = t_grid[seg_idx].item()
                t_right = t_grid[seg_idx+1].item()
                h = t_right - t_left
                
                # Use left endpoint time for all jumps in this segment
                t_jump = torch.tensor([t_left], device=device)
                
                # Model prediction at left endpoint
                p_1t = self.model(x=x_t, t=t_jump.repeat(batch_size), **model_extras)
                
                # Compute u_t(x,x_t)
                scheduler_output = self.path.scheduler(t=t_jump.repeat(batch_size))
                k_t = scheduler_output.alpha_t
                d_k_t = scheduler_output.d_alpha_t
                
                # Broadcast to (batch_size, dim, vocab_size)
                k_t = k_t.view(batch_size, 1, 1)
                d_k_t = d_k_t.view(batch_size, 1, 1)
                u = d_k_t / (1 - k_t) * p_1t
                
                # Set u_t(x_t,x_t) = 0
                delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

                # Batch Poisson sampling: sample n×D×S Poisson distributions
                # Expected value = u * h (transition rate × time interval)
                expected_jumps_tensor = u * h
                n_jumps_batch = torch.distributions.poisson.Poisson(expected_jumps_tensor).sample()
                
                # Tau-leaping in our paper
                # For each dimension, compute: x_new = x_current + Σ(x_target - x_current) * Poisson
                # Create target positions: (batch_size, dim, vocab_size)
                vocab_positions = torch.arange(self.vocabulary_size, device=device).view(1, 1, -1)
                vocab_positions = vocab_positions.expand(batch_size, x_t.shape[1], -1)
                
                # Current positions: (batch_size, dim) -> (batch_size, dim, 1)
                current_positions = x_t.unsqueeze(-1)
                
                # Compute (target - current) for each vocab: (batch_size, dim, vocab_size)
                position_diff = vocab_positions - current_positions
                
                # Multiply by Poisson samples: (batch_size, dim, vocab_size)
                increments = position_diff * n_jumps_batch
                
                # Sum over vocabulary dimension: (batch_size, dim)
                total_shift = increments.sum(dim=-1)
                x_current = x_t

                # Update: x_new = x_current + total_shift
                x_t = x_t + total_shift.long()
                
                # Mapping back to valid vocabulary range [0, vocabulary_size-1]
                out_of_range = (x_t < 0) | (x_t >= self.vocabulary_size)
                x_t = torch.where(out_of_range, x_current, x_t)

                # # Tau-leaping in Campbell et al. (2022)
                # # Batch Poisson sampling: sample n×D×S Poisson distributions
                # # Expected value = u * h (transition rate × time interval)
                # expected_jumps_tensor = u * h
                # n_jumps_batch = torch.distributions.poisson.Poisson(expected_jumps_tensor).sample()
                
                # # Sum over vocabulary dimension: (batch_size, dim, vocab_size) -> (batch_size, dim)
                # n_jumps_sum = n_jumps_batch.sum(dim=2)
                
                # # Only update dimensions where sum == 1
                # mask_update = (n_jumps_sum == 1)
                
                # if mask_update.any():
                #     # For dimensions where sum == 1, find which vocabulary got the jump
                #     new_values = torch.argmax(n_jumps_batch, dim=2)
                    
                #     # Update only the masked positions
                #     x_t = torch.where(mask_update, new_values, x_t)
                
                # Update progress after each segment
                if verbose:
                    ctx.update(1)
            if mask == True:
                mask_token = (x_t == self.vocabulary_size - 1)
                if mask_token.any() > 0:
                    t_jump_plus = torch.tensor([t_right], device=device)
                    # Get model prediction at final time
                    p_1t = self.model(x=x_t, t=t_jump_plus.repeat(batch_size), **model_extras)
                    
                    # Set probability of staying at current state to 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)
                    
                    # Sample new values from the distribution for mask positions
                    if mask_token.sum() > 0:
                        x_t[mask_token] = categorical(
                            p_1t[mask_token].to(dtype=dtype_categorical)
                        )
        
        return x_t, N
 


class MixtureDiscreteEulerSolver(Solver):
    """Euler solver for CTMC process defined by MixtureDiscreteProbPath"""
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        N: int = 100,
        delta: float = 0.05,
        opt_grid: bool = True,
        mask: bool = False,
        verbose: bool = False,
        dtype_categorical: torch.dtype = torch.float32,
        **model_extras,
    ) -> Tensor:
        """
        Euler sampling for CTMC.
        Args:
            x_init (Tensor): Initial state.
            t0 (float): Initial time.
            t1 (float): Final time.
            N (int): Number of segments in [0, 1-delta].
            delta (float): Early stopping threshold.
            opt_grid (bool): Whether to use optimized non-uniform grid. Defaults to True.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Extra model args.
        Returns:
            Tensor: Final state after segmented tau-leaping.
        
        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        device = x_init.device
        x_t = x_init.clone()
        batch_size, _ = x_t.shape
        
        if opt_grid:
            # Create grid: [1-delta^{(i-1)/N},1-delta^{i/N}], i=1,...,N
            t_grid = torch.tensor([1 - delta**(i/N) for i in range(0, N+1)], device=device)
        else:
            # Create uniform grid: [t0, t0+h, t0+2h, ..., t1-tau]
            # where h = (t1 - tau - t0) / N
            t_grid = torch.linspace(t0, t1 - delta, N + 1, device=device)

        
        
        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=N, desc=f"Euler")
        else:
            ctx = nullcontext()
        
        with ctx:
            for seg_idx in range(N):
                t_left = t_grid[seg_idx].item()
                t_right = t_grid[seg_idx+1].item()
                h = t_right - t_left
                
                # Use left endpoint time for all jumps in this segment
                t_jump = torch.tensor([t_left], device=device)
                
                # Model prediction at left endpoint
                p_1t = self.model(x=x_t, t=t_jump.repeat(batch_size), **model_extras)
                
                # Compute u_t(x,x_t)
                scheduler_output = self.path.scheduler(t=t_jump.repeat(batch_size))
                d_k_t = scheduler_output.d_alpha_t
                k_t = scheduler_output.alpha_t

                
                
                # Broadcast to (batch_size, dim, vocab_size)
                k_t = k_t.view(batch_size, 1, 1)
                d_k_t = d_k_t.view(batch_size, 1, 1)
                u = d_k_t / (1 - k_t) * p_1t * h


                # Set p_1t(x_t,x_t) = 0
                delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

                intensity = u.sum(dim=-1)
                mask_jump = torch.rand(
                    size=x_t.shape, device=x_t.device
                ) < 1 - torch.exp(-intensity)

                if mask_jump.sum() > 0:
                    x_t[mask_jump] = categorical(
                        u[mask_jump].to(dtype=dtype_categorical)
                    )
                
                # Update progress after each segment
                if verbose:
                    ctx.update(1)
            if mask == True:
                mask_token = (x_t == self.vocabulary_size - 1)
                if mask_token.any() > 0:
                    t_jump_plus = torch.tensor([t_right], device=device)
                    # Get model prediction at final time
                    p_1t = self.model(x=x_t, t=t_jump_plus.repeat(batch_size), **model_extras)
                    
                    # Set probability of staying at current state to 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)
                    
                    # Sample new values from the distribution for mask positions
                    if mask_token.sum() > 0:
                        x_t[mask_token] = categorical(
                            p_1t[mask_token].to(dtype=dtype_categorical)
                        )
        
        return x_t, N

class MixtureDiscreteTimeCorrectedSolver(Solver):
    """Time-corrected solver for CTMC process defined by MixtureDiscreteProbPath"""
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        N: int = 100,
        delta: float = 0.05,
        opt_grid: bool = True,
        mask: bool = False,
        verbose: bool = False,
        dtype_categorical: torch.dtype = torch.float32,
        **model_extras,
    ) -> Tensor:
        """
        Time-corrected sampling for CTMC.
        Args:
            x_init (Tensor): Initial state.
            t0 (float): Initial time.
            t1 (float): Final time.
            N (int): Number of segments in [0, 1-delta].
            delta (float): Early stopping threshold.
            opt_grid (bool): Whether to use optimized non-uniform grid. Defaults to True.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Extra model args.
        Returns:
            Tensor: Final state after segmented tau-leaping.
        
        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        device = x_init.device
        x_t = x_init.clone()
        batch_size, _ = x_t.shape
        
        if opt_grid:
            # Create grid: [1-delta^{(i-1)/N},1-delta^{i/N}], i=1,...,N
            t_grid = torch.tensor([1 - delta**(i/N) for i in range(0, N+1)], device=device)
        else:
            # Create uniform grid: [t0, t0+h, t0+2h, ..., t1-tau]
            # where h = (t1 - tau - t0) / N
            t_grid = torch.linspace(t0, t1 - delta, N + 1, device=device)

        
        
        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=N, desc=f"Time-corrected")
        else:
            ctx = nullcontext()
        
        with ctx:
            for seg_idx in range(N):
                t_left = t_grid[seg_idx].item()
                t_right = t_grid[seg_idx+1].item()
                h = t_right - t_left
                
                # Use left endpoint time for all jumps in this segment
                t_jump = torch.tensor([t_left], device=device)
                t_jump_plus = torch.tensor([t_right], device=device)
                
                # Model prediction at left endpoint
                p_1t = self.model(x=x_t, t=t_jump.repeat(batch_size), **model_extras)
                
                # Compute u_t(x,x_t)
                scheduler_output = self.path.scheduler(t=t_jump.repeat(batch_size))
                scheduler_output_plus = self.path.scheduler(t=t_jump_plus.repeat(batch_size))
                k_t = scheduler_output.alpha_t
                k_t_plus = scheduler_output_plus.alpha_t
                
                
                # Broadcast to (batch_size, dim, vocab_size)
                k_t = k_t.view(batch_size, 1, 1)
                k_t_plus = k_t_plus.view(batch_size, 1, 1)
                u = torch.log((1 - k_t) / (1 - k_t_plus)) * p_1t

                # Set p_1t(x_t,x_t) = 0
                delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

                intensity = u.sum(dim=-1)
                mask_jump = torch.rand(
                    size=x_t.shape, device=x_t.device
                ) < 1 - torch.exp(-intensity)

                if mask_jump.sum() > 0:
                    x_t[mask_jump] = categorical(
                        u[mask_jump].to(dtype=dtype_categorical)
                    )
                
                # Update progress after each segment
                if verbose:
                    ctx.update(1)
            if mask == True:
                mask_token = (x_t == self.vocabulary_size - 1)
                if mask_token.any() > 0:
                    t_jump_plus = torch.tensor([t_right], device=device)
                    # Get model prediction at final time
                    p_1t = self.model(x=x_t, t=t_jump_plus.repeat(batch_size), **model_extras)
                    
                    # Set probability of staying at current state to 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)
                    
                    # Sample new values from the distribution for mask positions
                    if mask_token.sum() > 0:
                        x_t[mask_token] = categorical(
                            p_1t[mask_token].to(dtype=dtype_categorical)
                        )

        return x_t, N
    


class MixtureDiscreteLocationCorrectedSolver(Solver):
    """Location-corrected solver for CTMC process defined by MixtureDiscreteProbPath"""
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size
        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        N: int = 100,
        delta: float = 0.05,
        opt_grid: bool = True,
        mask: bool = False,
        verbose: bool = False,
        dtype_categorical: torch.dtype = torch.float32,
        **model_extras,
    ) -> Tensor:
        """
        Time-corrected sampling for CTMC.
        Args:
            x_init (Tensor): Initial state.
            t0 (float): Initial time.
            t1 (float): Final time.
            N (int): Number of segments in [0, 1-delta].
            delta (float): Early stopping threshold.
            opt_grid (bool): Whether to use optimized non-uniform grid. Defaults to True.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Extra model args.
        Returns:
            Tensor: Final state after segmented tau-leaping.
        
        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        device = x_init.device
        x_t = x_init.clone()
        batch_size, _ = x_t.shape
        
        if opt_grid:
            # Create grid: [1-delta^{(i-1)/N},1-delta^{i/N}], i=1,...,N
            t_grid = torch.tensor([1 - delta**(i/N) for i in range(0, N+1)], device=device)
        else:
            # Create uniform grid: [t0, t0+h, t0+2h, ..., t1-tau]
            # where h = (t1 - tau - t0) / N
            t_grid = torch.linspace(t0, t1 - delta, N + 1, device=device)

        
        
        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=N, desc=f"Location-corrected")
        else:
            ctx = nullcontext()
        
        with ctx:
            for seg_idx in range(N):
                t_left = t_grid[seg_idx].item()
                t_right = t_grid[seg_idx+1].item()
                h = t_right - t_left
                
                # Use left endpoint time for all jumps in this segment
                t_jump = torch.tensor([t_left], device=device)
                t_jump_plus = torch.tensor([t_right], device=device)
                scheduler_output = self.path.scheduler(t=t_jump.repeat(batch_size))
                k_t = scheduler_output.alpha_t
                # Model prediction at left endpoint
                p_1t = self.model(x=x_t, t=t_jump.repeat(batch_size), **model_extras)
                delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)

                exp_intensity = p_1t.sum(dim=(1,2))
                exp_dist = torch.distributions.exponential.Exponential(exp_intensity).sample()
                exit_times = self.path.scheduler.kappa_inverse((1 - k_t) * (1 - torch.exp(-exp_dist)) + k_t)

                if (exit_times >= t_right).all():
                    continue

                mask_active = (exit_times < t_right)
                # p_1t shape: (batch_size, dim, vocab_size)
                # For samples that jump, flatten to (batch_size, dim * vocab_size)
                p_1t_flat = p_1t[mask_active].view(mask_active.sum(), -1)
                
                # Sample from flattened multinomial distribution
                # Returns indices in range [0, dim * vocab_size - 1]
                flat_indices = categorical(p_1t_flat.to(dtype=dtype_categorical))
                
                # Convert flat indices back to (dimension, value) pairs
                # flat_index = dimension * vocab_size + value
                sampled_dims = flat_indices // self.vocabulary_size  # which dimension
                sampled_values = flat_indices % self.vocabulary_size  # which vocab value
                
                # Get batch indices for samples that are active
                batch_indices = torch.arange(batch_size, device=device)[mask_active]
                
                # Update only the sampled dimension for each active sample
                x_t[batch_indices, sampled_dims] = sampled_values


                num_active = mask_active.sum()
                p_1t = self.model(x=x_t[mask_active], t=exit_times[mask_active], **model_extras)
                delta_t = F.one_hot(x_t[mask_active], num_classes=self.vocabulary_size)
                p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)
                scheduler_output_new = self.path.scheduler(t=exit_times[mask_active])
                k_t_new = scheduler_output_new.alpha_t
                scheduler_output_plus = self.path.scheduler(t=t_jump_plus.repeat(num_active))
                k_t_plus = scheduler_output_plus.alpha_t     
                
                # Broadcast to (batch_size, dim, vocab_size)
                k_t_new = k_t_new.view(num_active, 1, 1)
                k_t_plus = k_t_plus.view(num_active, 1, 1)
                u = torch.log((1 - k_t_new) / (1 - k_t_plus)) * p_1t


                intensity = u.sum(dim=-1)
                mask_jump = torch.rand(
                    size=(num_active, x_t.shape[1]), device=x_t.device
                ) < 1 - torch.exp(-intensity)

                if mask_jump.sum() > 0:
                    x_t_active = x_t[mask_active].clone()
                    x_t_active[mask_jump] = categorical(
                        u[mask_jump].to(dtype=dtype_categorical)
                    )
                    x_t[mask_active] = x_t_active
                
                # Update progress after each segment
                if verbose:
                    ctx.update(1)
            if mask == True:
                mask_token = (x_t == self.vocabulary_size - 1)
                if mask_token.any() > 0:
                    t_jump_plus = torch.tensor([t_right], device=device)
                    # Get model prediction at final time
                    p_1t = self.model(x=x_t, t=t_jump_plus.repeat(batch_size), **model_extras)
                    
                    # Set probability of staying at current state to 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    p_1t = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(p_1t), p_1t)
                    
                    # Sample new values from the distribution for mask positions
                    if mask_token.sum() > 0:
                        x_t[mask_token] = categorical(
                            p_1t[mask_token].to(dtype=dtype_categorical)
                        )
        
        return x_t, N