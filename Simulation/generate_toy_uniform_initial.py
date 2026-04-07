"""
draw_toy_gpu_uniform.py
Uniform source experiment with same-seed fairness across all samplers.
Outputs: tv_time_all_results_uniform.csv
"""

import os
import time
import random
import csv

import torch
import numpy as np

from dataset.dataset import generate_3k_discrete_data
from model_toy import ToyMLP
from utils import WrappedModel
from discrete_solver import (
    MixtureDiscreteTauleapingSolver,
    MixtureDiscreteEulerSolver,
    MixtureDiscreteTimeCorrectedSolver,
    MixtureDiscreteLocationCorrectedSolver,
    MixtureDiscreteRK2Solver,
    MixtureDiscreteRK2TrapezoidSolver,
)

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler


# ============================================================================
# RNG state helpers for same-seed fairness
# ============================================================================
def save_rng_state(device):
    state = {
        "torch_cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    torch.random.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


# ============================================================================
# Helpers
# ============================================================================
def total_variation(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def joint_to_index(joint_data, vocab_size=8):
    return joint_data[:, 0] * (vocab_size ** 2) + joint_data[:, 1] * vocab_size + joint_data[:, 2]


def run_sampler(sampler_name, posterior_model, path, vocab_size, x_init, N_step, delta):
    if sampler_name == "Tauleaping":
        solver = MixtureDiscreteTauleapingSolver(
            model=posterior_model, path=path, vocabulary_size=vocab_size)
        return solver.sample(x_init=x_init, N=N_step, delta=delta, verbose=False)
    elif sampler_name == "Euler":
        solver = MixtureDiscreteEulerSolver(
            model=posterior_model, path=path, vocabulary_size=vocab_size)
        return solver.sample(x_init=x_init, N=N_step, delta=delta, verbose=False)
    elif sampler_name == "TimeCorrected":
        solver = MixtureDiscreteTimeCorrectedSolver(
            model=posterior_model, path=path, vocabulary_size=vocab_size)
        return solver.sample(x_init=x_init, N=N_step, delta=delta, verbose=False)
    elif sampler_name == "LocationCorrected":
        solver = MixtureDiscreteLocationCorrectedSolver(
            model=posterior_model, path=path, vocabulary_size=vocab_size)
        return solver.sample(x_init=x_init, N=N_step, delta=delta, verbose=False)
    elif sampler_name == "RK2":
        solver = MixtureDiscreteRK2Solver(
            model=posterior_model, path=path, vocabulary_size=vocab_size)
        return solver.sample(x_init=x_init, N=N_step, delta=delta, theta=0.5, verbose=False)
    elif sampler_name == "RK2Trapezoid":
        solver = MixtureDiscreteRK2TrapezoidSolver(
            model=posterior_model, path=path, vocabulary_size=vocab_size)
        return solver.sample(x_init=x_init, N=N_step, delta=delta, theta=0.5, verbose=False)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


# ============================================================================
# Main
# ============================================================================
def main():
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Using gpu")
    else:
        device = "cpu"
        print("Using cpu.")

    # Config
    vocab_size = 8
    delta = 0.05
    n_samples_eval = 100000
    n_repeats = 100
    k_values = [1, 2, 3, 4, 5]
    N_values = [1,2,3,4,5,6,7,8,9, 10, 15, 20, 30, 40, 50, 75, 100]
    samplers = ["Tauleaping", "Euler", "TimeCorrected", "LocationCorrected", "RK2", "RK2Trapezoid"]

    scheduler = PolynomialConvexScheduler(n=1)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    joint_space_size = vocab_size ** 3

    print("\n" + "=" * 80)
    print("Uniform source experiment")
    print("=" * 80)

    tv_results = {k: {name: {} for name in samplers} for k in k_values}

    for k in k_values:
        dimension = 3 * k

        # Ground truth (fixed seed for reproducibility)
        real_data_cache = generate_3k_discrete_data(n=1000000, K=k, seed=0)
        real_joint = real_data_cache[:, :3]
        real_indices = joint_to_index(real_joint, vocab_size)
        unique_values_real, counts_real = np.unique(real_indices, return_counts=True)
        probs_real_full = np.zeros(joint_space_size)
        for val, prob in zip(unique_values_real, counts_real / len(real_indices)):
            probs_real_full[int(val)] = prob

        # Load model
        logit_model = ToyMLP(vocab_size=vocab_size, hidden_dim=256, length=dimension)
        model_path = os.path.join("./ckpts", f"toy_uniform_{k}_step200000", "ckpt.pth")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            continue
        ckpt = torch.load(model_path, map_location=device)
        logit_model.load_state_dict(ckpt)
        logit_model.to(device)
        logit_model.eval()
        posterior_model = WrappedModel(logit_model)

        for N_step in N_values:
            for rep in range(n_repeats):
                # Deterministic seed per (k, rep) only — same across all N_step
                # so that the same random samples are used for fair comparison
                seed = 42 + k * 250000 + rep * 2500
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)

                # Save RNG state — shared across all samplers
                rng_state = save_rng_state(device)

                for sampler_name in samplers:
                    # Restore RNG so every sampler sees the same x_init
                    restore_rng_state(rng_state, device)

                    x_init = torch.randint(
                        size=(n_samples_eval, dimension), high=vocab_size, device=device
                    )

                    with torch.no_grad():
                        start_time = time.time()
                        sol, _ = run_sampler(
                            sampler_name, posterior_model, path,
                            vocab_size, x_init, N_step, delta,
                        )
                        sample_time = time.time() - start_time

                    sol_joint = sol[:, :3].cpu().numpy()
                    sol_indices = joint_to_index(sol_joint, vocab_size)
                    unique_values_sol, counts_sol = np.unique(sol_indices, return_counts=True)
                    probs_sol_full = np.zeros(joint_space_size)
                    for val, prob in zip(unique_values_sol, counts_sol / len(sol_indices)):
                        probs_sol_full[int(val)] = prob

                    tv = total_variation(probs_real_full, probs_sol_full)

                    # Accumulate
                    if N_step not in tv_results[k][sampler_name]:
                        tv_results[k][sampler_name][N_step] = {"all_tv": [], "all_time": []}
                    tv_results[k][sampler_name][N_step]["all_tv"].append(tv)
                    tv_results[k][sampler_name][N_step]["all_time"].append(sample_time)


            # Print summary after all repeats for this (k, N_step)
            print(f"\n  --- Summary: k={k}, N={N_step} ---")
            for sampler_name in samplers:
                res = tv_results[k][sampler_name].get(N_step)
                if res:
                    print(
                        f"    {sampler_name}: TV={np.mean(res['all_tv']):.6f}±{np.std(res['all_tv']):.6f}, "
                        f"Time={np.mean(res['all_time']):.2f}±{np.std(res['all_time']):.2f}s",
                        flush=True,
                    )
            print()

    # Save CSV
    csv_filename = "tv_time_all_results_uniform_initial.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["k", "dimension", "N", "sampler", "tv_mean", "tv_std", "time_mean", "time_std"])
        for k in k_values:
            for N in N_values:
                for sampler_name in samplers:
                    res = tv_results[k][sampler_name].get(N)
                    if res:
                        writer.writerow([
                            k, 3 * k, N, sampler_name,
                            np.mean(res["all_tv"]), np.std(res["all_tv"]),
                            np.mean(res["all_time"]), np.std(res["all_time"]),
                        ])
                    else:
                        writer.writerow([k, 3 * k, N, sampler_name, None, None, None, None])

    print(f"{'=' * 80}")
    print(f"CSV saved: {csv_filename}")
    print(f"{'=' * 80}")
    print("Done!")


if __name__ == "__main__":
    main()
