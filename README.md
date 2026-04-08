# Corrected Samplers for Discrete Flow Models

PyTorch implementation of **Corrected Samplers for Discrete Flow Models**.

## Simulation

Toy simulation experiments.

```bash
cd Simulation
conda create -n simulation python=3.10 -y
conda activate simulation
pip install -r requirements.txt

# Train
python train_uniform_toy.py --device cuda
python train_mask_toy.py --device cuda

# Generate
python generate_toy_uniform_initial.py
python generate_toy_mask_initial.py
```
## Image Generation — Quick Start

```bash
cd ImageGen
conda create -n fudoki python=3.10 -y
conda activate fudoki
pip install -r requirements.txt

# Sampling
bash inference_t2i.sh
```

Model checkpoints are **auto-downloaded** on the first run.

## Acknowledgments

We gratefully acknowledge the following projects for providing code:

- [FUDOKI](https://github.com/fudoki-hku/FUDOKI)
- [flow_matching](https://github.com/facebookresearch/flow_matching)
- [DiscreteFastSolver](https://github.com/yuchen-zhu-zyc/DiscreteFastSolver)


## Citation

If this work is helpful to your research, you may cite our paper as follows.

```bibtex
@article{wan2026correctedsamplersdiscreteflow,
      title={Corrected Samplers for Discrete Flow Models}, 
      author={Zhengyan Wan and Yidong Ouyang and Liyan Xie and Fang Fang and Hongyuan Zha and Guang Cheng},
      year={2026},
      eprint={2601.22519},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2601.22519}, 
}
```
