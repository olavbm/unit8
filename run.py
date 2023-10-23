import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import tempfile
import json
import shutil
import imageio

from wasabi import Printer
from util import parse_args

msg = Printer()
args = parse_args()

class Agent(nn.Module):
    def __init__(self):
        super().__init__()

    def act(self, state):
        pass

def make_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)


def train_agent():
    agent = Agent()
    env = make_env(args.env_id, 1337)
    return agent, env

def foo():
    return train_agent()

def test_this_is_foo():
    assert foo() == 2
