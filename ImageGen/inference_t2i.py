import os
import argparse
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import random
import time

# Patch torch.load for PyTorch >= 2.6 (weights_only default changed to True)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from fudoki.eval_loop import CFGScaledModel
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver.discrete_solver import (
    MixtureDiscreteSoftmaxEulerSolver,
    MixtureDiscreteTimeCorrectedSolver,
    MixtureDiscreteLocationCorrectedSolver,
    MixtureDiscreteSoftmaxRK2Solver,
    MixtureDiscreteSoftmaxRK2TrapezoidSolver,
)
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model

VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576
CFG_SCALE = 5.0

PROMPT = "A beautiful modern wooden house, close to the lake, in the mountains at sunrise, anime style"

HF_REPO_ID = "LucasJinWang/FUDOKI"


def ensure_checkpoints(ckpt_path: str) -> str:
    """If checkpoint dir is missing or empty, auto-download from HuggingFace."""
    ckpt = Path(ckpt_path)
    if ckpt.is_dir() and any(ckpt.iterdir()):
        print(f"[ok] Checkpoints found: {ckpt}")
        return str(ckpt)
    print(f"[!] Checkpoints not found at {ckpt}, downloading from HuggingFace...")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
        subprocess.run(["git", "clone", f"https://huggingface.co/{HF_REPO_ID}", str(ckpt)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO_ID, local_dir=str(ckpt))
    print(f"[ok] Checkpoints downloaded to: {ckpt}")
    return str(ckpt)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Single-prompt T2I inference with corrected samplers.")
    parser.add_argument("--seed", type=int, default=666, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--text_embedding_path", type=str, required=True, help="Path to the text embedding.")
    parser.add_argument("--image_embedding_path", type=str, required=True, help="Path to the image embedding.")
    parser.add_argument("--discrete_fm_steps", type=int, default=8, help="Inference steps for discrete flow matching.")
    parser.add_argument("--txt_max_length", type=int, default=500, help="Text length maximum.")
    parser.add_argument("--output_dir", type=str, default="./output_t2i", help="Directory to save the output files.")
    return parser.parse_args()


def build_data_info(prompt, batch_size, vl_chat_processor, txt_max_length, device):
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    sft_format = sft_format + vl_chat_processor.image_start_tag
    input_ids = vl_chat_processor.tokenizer.encode(sft_format)
    input_ids = torch.LongTensor(input_ids)
    img_start = input_ids.shape[0]
    input_ids = torch.cat([
        input_ids,
        torch.LongTensor([vl_chat_processor.image_id] * IMG_LEN),
        torch.LongTensor([vl_chat_processor.image_end_id]),
    ])
    img_end = input_ids.shape[0] - 1

    original_input_id_len = input_ids.shape[0]
    if original_input_id_len >= txt_max_length + IMG_LEN:
        raise ValueError("Sentences too long, not supported so far...")

    rows_to_pad = txt_max_length + IMG_LEN - input_ids.shape[0]
    input_ids = torch.cat([
        input_ids,
        torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad),
    ], dim=0)
    attention_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    attention_mask[:original_input_id_len] = True

    image_expanded_token_mask = torch.zeros_like(input_ids)
    image_expanded_token_mask[img_start:img_end] = True

    text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
    split_token = vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    split_token_length = len(split_token)
    start_index = -1
    for j in range(len(input_ids) - split_token_length + 1):
        if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
            start_index = j
            break
    if start_index == -1:
        raise ValueError("Split token not found in input_ids")
    text_expanded_token_mask[1:(start_index + split_token_length)] = 1

    data_info = {
        "text_token_mask": text_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device),
        "image_token_mask": image_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device),
        "generation_or_understanding_mask": torch.Tensor([1]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device),
        "attention_mask": attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device),
        "sft_format": sft_format,
        "understanding_img": torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device),
        "has_understanding_img": torch.Tensor([False]).to(dtype=int).repeat(batch_size).to(device),
    }

    input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(batch_size, 1).to(device)
    x_0_img = torch.randint(16384, input_ids.shape, dtype=torch.long, device=device)
    x_init = x_0_img * data_info["image_token_mask"] + input_ids * (1 - data_info["image_token_mask"])

    return x_init, data_info


def decode_and_save(model, synthetic_samples, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    synthetic_samples = model.gen_vision_model.decode_code(
        synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]
    )
    synthetic_samples = torch.clamp((synthetic_samples + 1) / 2.0, 0.0, 1.0)
    image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for k in range(image.shape[0]):
        path = os.path.join(save_dir, f"{prefix}_{k:04d}.png")
        Image.fromarray(image[k]).save(path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    args = parse_arguments()

    # Resolve paths to absolute (transformers requires absolute paths)
    args.checkpoint_path = str(Path(args.checkpoint_path).resolve())
    args.text_embedding_path = str(Path(args.text_embedding_path).resolve())
    args.image_embedding_path = str(Path(args.image_embedding_path).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())

    # Auto-download checkpoints if missing
    args.checkpoint_path = ensure_checkpoints(args.checkpoint_path)
    args.text_embedding_path = os.path.join(args.checkpoint_path, "text_embedding.pt")
    args.image_embedding_path = os.path.join(args.checkpoint_path, "image_embedding.pt")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    device = "cuda"

    print(f"Prompt: {PROMPT}")
    print(f"Seed: {seed}, Steps: {args.discrete_fm_steps}, Batch: {args.batch_size}")

    # Load model
    model = instantiate_model(args.checkpoint_path).to(device).to(torch.float32)
    model.train(False)
    vl_chat_processor = VLChatProcessor.from_pretrained(args.checkpoint_path)
    cfg_weighted_model = CFGScaledModel(model=model, g_or_u="generation")

    batch_size = args.batch_size
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    discrete_fm_steps = args.discrete_fm_steps

    with torch.no_grad():
        path_txt = MixtureDiscreteSoftmaxProbPath(mode="text", embedding_path=args.text_embedding_path)
        path_img = MixtureDiscreteSoftmaxProbPath(mode="image", embedding_path=args.image_embedding_path)

        x_init, data_info = build_data_info(
            PROMPT, batch_size, vl_chat_processor, args.txt_max_length, device
        )

        # Save RNG states so all solvers start from the same initial noise
        rng_state_cpu = torch.random.get_rng_state()
        rng_state_cuda = torch.cuda.get_rng_state(device)
        rng_state_np = np.random.get_state()
        rng_state_py = random.getstate()

        def restore_rng():
            torch.random.set_rng_state(rng_state_cpu)
            torch.cuda.set_rng_state(rng_state_cuda, device)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)

        # ---- 1. Euler Sampler ----
        restore_rng()
        print(f"\n[1/5] Euler sampler (steps={discrete_fm_steps})...")
        t0 = time.time()
        solver = MixtureDiscreteSoftmaxEulerSolver(
            model=cfg_weighted_model,
            path_txt=path_txt, path_img=path_img,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )
        samples = solver.sample(
            x_init=x_init, step_size=1.0 / discrete_fm_steps,
            verbose=True, return_intermediates=False, div_free=0,
            dtype_categorical=torch.float32, datainfo=data_info, cfg_scale=CFG_SCALE,
        )
        dt = time.time() - t0
        print(f"  Time: {dt:.2f}s")
        decode_and_save(model, samples, save_path, f"euler_{discrete_fm_steps}steps")

        # ---- 2. RK2 Sampler ----
        restore_rng()
        rk2_steps = discrete_fm_steps // 2
        print(f"\n[2/5] RK2 sampler (steps={rk2_steps})...")
        t0 = time.time()
        solver = MixtureDiscreteSoftmaxRK2Solver(
            model=cfg_weighted_model,
            path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )
        samples = solver.sample(
            x_init=x_init, step_size=2.0 / discrete_fm_steps, theta=0.5,
            verbose=True, return_intermediates=False,
            dtype_categorical=torch.float32, datainfo=data_info, cfg_scale=CFG_SCALE,
        )
        dt = time.time() - t0
        print(f"  Time: {dt:.2f}s")
        decode_and_save(model, samples, save_path, f"rk2_{rk2_steps}steps")

        # ---- 3. RK2 Trapezoid Sampler ----
        restore_rng()
        print(f"\n[3/5] RK2 Trapezoid sampler (steps={rk2_steps})...")
        t0 = time.time()
        solver = MixtureDiscreteSoftmaxRK2TrapezoidSolver(
            model=cfg_weighted_model,
            path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )
        samples = solver.sample(
            x_init=x_init, step_size=2.0 / discrete_fm_steps, theta=0.5,
            verbose=True, return_intermediates=False,
            dtype_categorical=torch.float32, datainfo=data_info, cfg_scale=CFG_SCALE,
        )
        dt = time.time() - t0
        print(f"  Time: {dt:.2f}s")
        decode_and_save(model, samples, save_path, f"rk2trap_{rk2_steps}steps")

        # ---- 4. Time-Corrected Sampler ----
        restore_rng()
        print(f"\n[4/5] Time-Corrected sampler (steps={discrete_fm_steps})...")
        t0 = time.time()
        solver = MixtureDiscreteTimeCorrectedSolver(
            model=cfg_weighted_model,
            path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )
        samples = solver.sample(
            x_init=x_init, step_size=1.0 / discrete_fm_steps,
            verbose=True, return_intermediates=False, m=32,
            dtype_categorical=torch.float32, datainfo=data_info, cfg_scale=CFG_SCALE,
        )
        dt = time.time() - t0
        print(f"  Time: {dt:.2f}s")
        decode_and_save(model, samples, save_path, f"timecorrected_{discrete_fm_steps}steps")

        # ---- 5. Location-Corrected Sampler ----
        restore_rng()
        lc_steps = discrete_fm_steps // 2
        print(f"\n[5/5] Location-Corrected sampler (steps={lc_steps})...")
        t0 = time.time()
        solver = MixtureDiscreteLocationCorrectedSolver(
            model=cfg_weighted_model,
            path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )
        samples = solver.sample(
            x_init=x_init, step_size=2.0 / discrete_fm_steps,
            verbose=True, return_intermediates=False,
            m=32, j=int(576 // (2 * discrete_fm_steps / 2)), t_theta=0,
            dtype_categorical=torch.float32, datainfo=data_info, cfg_scale=CFG_SCALE,
        )
        dt = time.time() - t0
        print(f"  Time: {dt:.2f}s")
        decode_and_save(model, samples, save_path, f"locationcorrected_{lc_steps}steps")

    print(f"\nAll done! Images saved to: {save_path}")
