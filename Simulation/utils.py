import argparse
import numpy as np
import torch
from torch import nn, Tensor
from flow_matching.path.scheduler.scheduler import SchedulerOutput, ConvexScheduler
from flow_matching.utils import ModelWrapper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all") # OpenAI gym environment name
    parser.add_argument("--expid", default="toy")  # Experiment ID
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--device", default="cpu", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--schedule', type=str, default="Linear")  
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args


class WrappedModel(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)

class LogitsModelWrapper(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.log(self.model(x, t) + 1e-16)

