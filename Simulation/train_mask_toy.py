import os
import tqdm
import functools
import ipdb
import torch
from torch import nn, Tensor

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from model_toy import ToyMLP
from utils import get_args
from dataset.dataset import *

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

def tau_to_str(tau_value):
    """
    tau_value -> str
    e.g. 0.001 -> '001', 0.01 -> '01', 0.1 -> '1', 1.0 -> '1'
    """
    tau_str = str(tau_value)
    if '.' in tau_str:
        decimal_part = tau_str.split('.')[1]
        decimal_part = decimal_part.rstrip('0')
        return decimal_part if decimal_part else '0'
    else:
        return tau_str

def train(args, pretrained_model, info, start_step=0):
    optimizer = Adam(pretrained_model.parameters(), lr=1e-3)
    scheduler = PolynomialConvexScheduler(n=1)  # linear scheduler
    path = MixtureDiscreteProbPath(scheduler=scheduler) # mixture discrete path
    loss_fn = MixturePathGeneralizedKL(path=path) # loss function

    n_steps = 200000
    batch_size = 512
    

    # info
    delta = 0.05
    vocab_size = info["vocab_size"]
    mask_token = info["mask_token"]
    k = info["K"]
    tqdm_step = tqdm.trange(start_step, n_steps)
    for step in tqdm_step:
        np_data = generate_3k_discrete_data(n=batch_size,K=k)
        x = torch.from_numpy(np_data).long().to(args.device)

        # x shape: (B, l)
        x_0 = torch.zeros_like(x) + mask_token
        t = torch.rand(x.shape[0]).to(args.device) * (1 - delta)
        # sample probability path
        path_sample = path.sample(t=t, x_0=x_0, x_1=x)
        logits = pretrained_model(x=path_sample.x_t, t=path_sample.t)
        loss = loss_fn(logits=logits, x_1=x, x_t=path_sample.x_t, t=path_sample.t)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()

        tqdm_step.set_description('Average Loss: {:5f}'.format(loss.item()))
        # Update the checkpoint after each epoch of training.
        if step % 5000 == 4999 and args.save_model:
            save_dir = os.path.join("./ckpts", "toy_mask_{}_step{}".format(k, step+1))
            os.makedirs(save_dir, exist_ok=True) 
            torch.save(pretrained_model.state_dict(), os.path.join(save_dir, "ckpt.pth"))
        wandb.log({"loss": loss.item(), "step": step, "k": k})

def main(args):
    for dir in ["./ckpts", "./toylogs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./ckpts", "toy_mask_{}".format(1e5))):
        os.makedirs(os.path.join("./ckpts", "toy_mask_{}".format(1e5)))

    # n_values = [100000]
    
    wandb.init(project="toy_loss_monitoring", name="toy_continue_training_multi_d")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mask_token = 8
    vocab_size = 9

    
    for k in [1, 2, 3, 4, 5]:
        info = {
            "vocab_size": vocab_size,
            "K": k,
            "mask_token": mask_token
        }
        
        checkpoint_path = os.path.join("./ckpts", f"toy_mask_{k}_step200000", f"ckpt.pth")
        start_step = 0
        # no time embedding for masked source distribution
        pretrained_model = ToyMLP(vocab_size=vocab_size, hidden_dim=256, length=3*k, time_dim=0).to(args.device)
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            pretrained_model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            start_step = 100000
            print(f"Resuming training for K={k} from step {start_step}...")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            print(f"Starting training from scratch for K={k}...")
        
        
        train(args, pretrained_model, info, start_step=start_step)
        print(f"Finished training for K={k}\n")

if __name__ == "__main__":
    args = get_args()
    main(args)