# Simulation — Quick Start

## Setup

```bash
cd Simulation
conda create -n simulation python=3.10 -y
conda activate simulation
pip install -r requirements.txt
```

## Train

```bash
# Uniform initial distribution
python train_uniform_toy.py --device cuda

# Mask initial distribution
python train_mask_toy.py --device cuda
```

## Generate & Evaluate

```bash
python generate_toy_uniform_initial.py
python generate_toy_mask_initial.py
```

Results are saved to `tv_time_all_results_uniform_initial.csv` and `tv_time_all_results_mask_initial.csv`.
