# Image Generation

Image generation with corrected samplers on [FUDOKI](https://huggingface.co/LucasJinWang/FUDOKI).

## Setup

```bash
cd ImageGen
conda create -n fudoki python=3.10 -y
conda activate fudoki
pip install -r requirements.txt
```

## Quick Start

```bash
bash inference_t2i.sh
```

Checkpoints are **auto-downloaded** on the first run. Output images are saved to `./output_t2i/`.


### GenEval

```bash
bash inference_t2i_geneval.sh
```

