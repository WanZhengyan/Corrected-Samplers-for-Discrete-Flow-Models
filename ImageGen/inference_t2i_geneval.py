import os
import argparse
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.distributed as dist
from torchvision import transforms
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
import random
import time
import pandas as pd

# Patch torch.load for PyTorch >= 2.6 (weights_only default changed to True)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from fudoki.eval_loop import CFGScaledModel
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver.discrete_solver import MixtureDiscreteSoftmaxEulerSolver, MixtureDiscreteTimeCorrectedSolver, MixtureDiscreteLocationCorrectedSolver, MixtureDiscreteSoftmaxRK2Solver, MixtureDiscreteSoftmaxRK2TrapezoidSolver
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model

import json

VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576
CFG_SCALE = 5.0

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
    parser = argparse.ArgumentParser(description="Run the script with custom arguments.")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--checkpoint_path", type=str, default="./pretrained_model", help="Path to the checkpoint directory.")
    parser.add_argument("--text_embedding_path", type=str, default="./text_embedding.pt", help="Path to the text embedding.")
    parser.add_argument("--image_embedding_path", type=str, default="./image_embedding.pt", help="Path to the image embedding.")
    parser.add_argument("--discrete_fm_steps", type=int, default=8, help="Inference steps for discrete flow matching")
    parser.add_argument("--txt_max_length", type=int, default=500, help="Text length maximum")
    parser.add_argument("--output_dir", type=str, default="./output_geneval", help="Directory to save the output files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Resolve paths to absolute
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
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    print('world_size', world_size)
    print('local_rank', local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    device = 'cuda'

    model_path = args.checkpoint_path
    model = instantiate_model(
       model_path
    ).to(device).to(torch.float32)
    model.train(False)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    batch_size = args.batch_size
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    discrete_fm_steps = args.discrete_fm_steps
    txt_max_length = args.txt_max_length
        
    cfg_weighted_model = CFGScaledModel(model=model, g_or_u='generation') 

    sampling_times_euler = []
    sampling_times_timecorrected = []
    sampling_times_locationcorrected = []
    sampling_times_rk2 = []
    sampling_times_rk2trapezoid = []

    with torch.no_grad():
        path_txt = MixtureDiscreteSoftmaxProbPath(mode='text', embedding_path=args.text_embedding_path)
        path_img = MixtureDiscreteSoftmaxProbPath(mode='image', embedding_path=args.image_embedding_path)



        generation_understanding_indicator = 1 # is generation mode if 1, understanding mode if 0
        EVAL_METADATA_FILE = "./evaluation_metadata.jsonl"
        with open(EVAL_METADATA_FILE, "r") as f:
            metadata_lines = f.readlines()
        with open('./geneval_prompt.txt', "r") as f:
            validation_prompts = f.read().splitlines()
        for prompt_index, prompt in enumerate(validation_prompts):
            prompt_seed = seed + prompt_index
            random.seed(prompt_seed)
            np.random.seed(prompt_seed)
            torch.manual_seed(prompt_seed)
            torch.cuda.manual_seed_all(prompt_seed)

            img = None
            conversation = [ # Multi-turn conversation
                {
                    "role": "User",
                    "content": prompt#"A beautiful modern wooden house, close to the lake, in the mountains at sunrise, anime style"
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            # output: str
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            sft_format = sft_format + vl_chat_processor.image_start_tag # concat image start tag, output: str
            input_ids = vl_chat_processor.tokenizer.encode(sft_format) # tokenizer encoding output: list(sequence of int)
            input_ids = torch.LongTensor(input_ids) # tensor conversion output: tensor; shape: (length)
            img_start = input_ids.shape[0]
            # concat image token ids; input_ids length: length + IMG_LEN + 1
            input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.image_id]*IMG_LEN), torch.LongTensor([vl_chat_processor.image_end_id])])
            img_end = input_ids.shape[0] - 1

            # pad tokens
            original_input_id_len = input_ids.shape[0]
        
            if original_input_id_len >= txt_max_length + IMG_LEN:
                raise ValueError("Sentences too long, not supported so far...")
            else:
                rows_to_pad = txt_max_length+IMG_LEN-input_ids.shape[0]
                # pad input_ids to txt_max_length + IMG_LEN
                input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
                attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
                attention_mask[:original_input_id_len] = True # True for original input ids, False for padded ids
            
            
            # obtain image token mask and fill in img token_ids
            image_expanded_token_mask = torch.zeros_like(input_ids)
            image_expanded_token_mask[img_start: img_end] = True # True for image token ids, False for else

            # obtain text token mask
            # We assume that there is only one turn for assistant to respond
            text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
            split_token = vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False) # output: list(sequence of int)
            split_token_length = len(split_token) # length of split token
            
            start_index = -1
            for j in range(len(input_ids) - split_token_length + 1):
                if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
                    start_index = j
                    break
            # start_index is the index of the first token of "Assistant:" in input_ids
            if start_index != -1:
                text_expanded_token_mask[1: (start_index+split_token_length)] = 1
            else:
                raise ValueError("Split token not found in input_ids")

                    
            # data_info includes "text_token_mask", "image_token_mask", "generation_or_understanding_mask", "attention_mask", "sft_format"
            # and "understanding_img" (if generation_or_understanding_mask is 0)
            generation_or_understanding_mask = generation_understanding_indicator
            data_info = dict()
            data_info['text_token_mask'] = text_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
            data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
            data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)

            data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
            data_info['sft_format'] = sft_format
            if generation_or_understanding_mask == 1:
                # all zero shape: (batch_size, 3, 384, 384)
                data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).repeat(batch_size).to(device)
            else:
                if img is not None:
                    # img is the understanding image, shape: (3, 384, 384)
                    data_info['understanding_img'] = img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                    data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int).repeat(batch_size).to(device)
                else:
                    # if img is None, we use a zero tensor as the understanding image
                    data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                    data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).repeat(batch_size).to(device)
            


            # shape: (batch_size, length)
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(batch_size, 1).to(device)

            # x_0_img is the initial image token ids, which is uniformly sampled from the vocabulary
            x_0_img = torch.randint(16384, input_ids.shape, dtype=torch.long, device=device)
            # data_info['image_token_mask'] is a mask that indicates which tokens are image tokens
            x_init = x_0_img * data_info['image_token_mask'] + input_ids * (1 - data_info['image_token_mask'])
            
            # Save RNG states so all solvers start from the same random state
            rng_state_cpu = torch.random.get_rng_state()
            rng_state_cuda = torch.cuda.get_rng_state(device)
            rng_state_np = np.random.get_state()
            rng_state_py = random.getstate()
            
            # Euler sampler
            torch.cuda.synchronize()
            torch.random.set_rng_state(rng_state_cpu)
            torch.cuda.set_rng_state(rng_state_cuda, device)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)
            start_time = time.time()
            solver = MixtureDiscreteSoftmaxEulerSolver(
                model=cfg_weighted_model,
                path_txt=path_txt,
                path_img=path_img,
                vocabulary_size_txt=VOCABULARY_SIZE_TXT,
                vocabulary_size_img=VOCABULARY_SIZE_IMG
            )
            synthetic_samples = solver.sample(
                x_init=x_init,
                step_size=1.0/discrete_fm_steps,
                verbose=True,
                return_intermediates=False,
                div_free=0,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=CFG_SCALE
            )
            end_time = time.time()
            sampling_times_euler.append(end_time - start_time)
            # input: (batch_size,sequence_length) -> output: (batch_size, 8, 24, 24)
            synthetic_samples = model.gen_vision_model.decode_code(synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]) # output value is between [-1, 1]
            synthetic_samples = (synthetic_samples + 1) / 2.0 # normalize to [0, 1]
            synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0) # clamp the values to [0, 1]
            image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            dir_name = f"{prompt_index:05d}"
            save_dir = os.path.join(save_path, f"Euler{discrete_fm_steps}", dir_name)
            os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    

            sample_json = json.loads(metadata_lines[prompt_index])
            with open(os.path.join(save_dir, "metadata.jsonl"), "w") as fp:
                json.dump(sample_json, fp)

            for k in range(synthetic_samples.shape[0]):
                image_save_path = os.path.join(os.path.join(save_dir, "samples"), f"{k:04d}.png")
                Image.fromarray(image[k]).save(image_save_path)

            # RK2 sampler
            torch.cuda.synchronize()
            torch.random.set_rng_state(rng_state_cpu)
            torch.cuda.set_rng_state(rng_state_cuda, device)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)
            start_time = time.time()
            solver = MixtureDiscreteSoftmaxRK2Solver(
                model=cfg_weighted_model,
                path_img=path_img,
                vocabulary_size_img=VOCABULARY_SIZE_IMG
            )
            synthetic_samples = solver.sample(
                x_init=x_init,
                step_size=2.0/discrete_fm_steps,
                theta=0.5,
                verbose=True,
                return_intermediates=False,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=CFG_SCALE
            )
            end_time = time.time()
            sampling_times_rk2.append(end_time - start_time)
            # input: (batch_size,sequence_length) -> output: (batch_size, 8, 24, 24)
            synthetic_samples = model.gen_vision_model.decode_code(synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]) # output value is between [-1, 1]
            synthetic_samples = (synthetic_samples + 1) / 2.0 # normalize to [0, 1]
            synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0) # clamp the values to [0, 1]
            image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            dir_name = f"{prompt_index:05d}"
            save_dir = os.path.join(save_path, f"RK2{discrete_fm_steps//2}", dir_name)
            os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

            sample_json = json.loads(metadata_lines[prompt_index])
            with open(os.path.join(save_dir, "metadata.jsonl"), "w") as fp:
                json.dump(sample_json, fp)

            for k in range(synthetic_samples.shape[0]):
                image_save_path = os.path.join(os.path.join(save_dir, "samples"), f"{k:04d}.png")
                Image.fromarray(image[k]).save(image_save_path)


            # RK2Trapezoid sampler
            torch.cuda.synchronize()
            torch.random.set_rng_state(rng_state_cpu)
            torch.cuda.set_rng_state(rng_state_cuda, device)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)
            start_time = time.time()
            solver = MixtureDiscreteSoftmaxRK2TrapezoidSolver(
                model=cfg_weighted_model,
                path_img=path_img,
                vocabulary_size_img=VOCABULARY_SIZE_IMG
            )
            synthetic_samples = solver.sample(
                x_init=x_init,
                step_size=2.0/discrete_fm_steps,
                theta=0.5,
                verbose=True,
                return_intermediates=False,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=CFG_SCALE
            )
            end_time = time.time()
            sampling_times_rk2trapezoid.append(end_time - start_time)
            # input: (batch_size,sequence_length) -> output: (batch_size, 8, 24, 24)
            synthetic_samples = model.gen_vision_model.decode_code(synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]) # output value is between [-1, 1]
            synthetic_samples = (synthetic_samples + 1) / 2.0 # normalize to [0, 1]
            synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0) # clamp the values to [0, 1]
            image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            dir_name = f"{prompt_index:05d}"
            save_dir = os.path.join(save_path, f"RK2Trapezoid{discrete_fm_steps//2}", dir_name)
            os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

            sample_json = json.loads(metadata_lines[prompt_index])
            with open(os.path.join(save_dir, "metadata.jsonl"), "w") as fp:
                json.dump(sample_json, fp)

            for k in range(synthetic_samples.shape[0]):
                image_save_path = os.path.join(os.path.join(save_dir, "samples"), f"{k:04d}.png")
                Image.fromarray(image[k]).save(image_save_path)


            # TimeCorrected sampler
            torch.cuda.synchronize()
            torch.random.set_rng_state(rng_state_cpu)
            torch.cuda.set_rng_state(rng_state_cuda, device)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)
            start_time = time.time()
            solver = MixtureDiscreteTimeCorrectedSolver(
                model=cfg_weighted_model,
                path_img=path_img,
                vocabulary_size_img=VOCABULARY_SIZE_IMG
            )
            synthetic_samples = solver.sample(
                x_init=x_init,
                step_size=1.0/discrete_fm_steps,
                verbose=True,
                return_intermediates=False,
                m=32,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=CFG_SCALE
            )
            end_time = time.time()
            sampling_times_timecorrected.append(end_time - start_time)
            # input: (batch_size,sequence_length) -> output: (batch_size, 8, 24, 24)
            synthetic_samples = model.gen_vision_model.decode_code(synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]) # output value is between [-1, 1]
            synthetic_samples = (synthetic_samples + 1) / 2.0 # normalize to [0, 1]
            synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0) # clamp the values to [0, 1]
            image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            
            dir_name = f"{prompt_index:05d}"
            save_dir = os.path.join(save_path, f"TimeCorrected{discrete_fm_steps}", dir_name)
            os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    
            sample_json = json.loads(metadata_lines[prompt_index])
            with open(os.path.join(save_dir, "metadata.jsonl"), "w") as fp:
                json.dump(sample_json, fp)
    
            for k in range(synthetic_samples.shape[0]):
                image_save_path = os.path.join(os.path.join(save_dir, "samples"), f"{k:04d}.png")
                Image.fromarray(image[k]).save(image_save_path)


            # LocationCorrected sampler
            torch.cuda.synchronize()
            torch.random.set_rng_state(rng_state_cpu)
            torch.cuda.set_rng_state(rng_state_cuda, device)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)
            start_time = time.time()
            solver = MixtureDiscreteLocationCorrectedSolver(
                model=cfg_weighted_model,
                path_img=path_img,
                vocabulary_size_img=VOCABULARY_SIZE_IMG
            )
            synthetic_samples = solver.sample(
                x_init=x_init,
                step_size=2.0/discrete_fm_steps,
                verbose=True,
                return_intermediates=False,
                m=32,
                j=int(576//(2 * discrete_fm_steps / 2)),
                t_theta=0,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=CFG_SCALE
            )
            end_time = time.time()
            sampling_times_locationcorrected.append(end_time - start_time)

            # input: (batch_size,sequence_length) -> output: (batch_size, 8, 24, 24)
            synthetic_samples = model.gen_vision_model.decode_code(synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]) # output value is between [-1, 1]
            synthetic_samples = (synthetic_samples + 1) / 2.0 # normalize to [0, 1]
            synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0) # clamp the values to [0, 1]
            image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            

            dir_name = f"{prompt_index:05d}"
            save_dir = os.path.join(save_path, f"LocationCorrected{discrete_fm_steps//2}", dir_name)
            os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    
            sample_json = json.loads(metadata_lines[prompt_index])
            with open(os.path.join(save_dir, "metadata.jsonl"), "w") as fp:
                json.dump(sample_json, fp)
    
            for k in range(synthetic_samples.shape[0]):
                image_save_path = os.path.join(os.path.join(save_dir, "samples"), f"{k:04d}.png")
                Image.fromarray(image[k]).save(image_save_path)



    csv_path_euler = os.path.join(save_path, f'Euler{discrete_fm_steps}', 'sampling_times_euler.csv')
    avg_time_euler = sum(sampling_times_euler) / len(sampling_times_euler) if sampling_times_euler else 0
    df_euler = pd.DataFrame({'sampling_time': sampling_times_euler})
    df_euler.loc['average'] = [avg_time_euler]
    df_euler.to_csv(csv_path_euler, index=True)


    csv_path_rk2 = os.path.join(save_path, f'RK2{discrete_fm_steps//2}', 'sampling_times_rk2.csv')
    avg_time_rk2 = sum(sampling_times_rk2) / len(sampling_times_rk2) if sampling_times_rk2 else 0
    df_rk2 = pd.DataFrame({'sampling_time': sampling_times_rk2})
    df_rk2.loc['average'] = [avg_time_rk2]
    df_rk2.to_csv(csv_path_rk2, index=True)


    csv_path_rk2trapezoid = os.path.join(save_path, f'RK2Trapezoid{discrete_fm_steps//2}', 'sampling_times_rk2trapezoid.csv')
    avg_time_rk2trapezoid = sum(sampling_times_rk2trapezoid) / len(sampling_times_rk2trapezoid) if sampling_times_rk2trapezoid else 0
    df_rk2trapezoid = pd.DataFrame({'sampling_time': sampling_times_rk2trapezoid})
    df_rk2trapezoid.loc['average'] = [avg_time_rk2trapezoid]
    df_rk2trapezoid.to_csv(csv_path_rk2trapezoid, index=True)

    csv_path_timecorrected = os.path.join(save_path, f'TimeCorrected{discrete_fm_steps}', 'sampling_times_timecorrected.csv')
    avg_time_timecorrected = sum(sampling_times_timecorrected) / len(sampling_times_timecorrected) if sampling_times_timecorrected else 0
    df_timecorrected = pd.DataFrame({'sampling_time': sampling_times_timecorrected})
    df_timecorrected.loc['average'] = [avg_time_timecorrected]
    df_timecorrected.to_csv(csv_path_timecorrected, index=True)


    csv_path_locationcorrected = os.path.join(save_path, f'LocationCorrected{discrete_fm_steps//2}', 'sampling_times_locationcorrected.csv')
    avg_time_locationcorrected = sum(sampling_times_locationcorrected) / len(sampling_times_locationcorrected) if sampling_times_locationcorrected else 0
    df_locationcorrected = pd.DataFrame({'sampling_time': sampling_times_locationcorrected})
    df_locationcorrected.loc['average'] = [avg_time_locationcorrected]
    df_locationcorrected.to_csv(csv_path_locationcorrected, index=True)
