import collections
import numpy as np
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from wasabi import Printer

from util import parse_args

SEED = 1337
MEM_SIZE=2000
msg = Printer()
args = parse_args()

mem = collections.deque(maxlen=MEM_SIZE)

class Agent(nn.Module):
    def __init__(self, dim_obs, dim_actions):
        super().__init__()
        self.layer1 = nn.Linear(dim_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, dim_actions)

    def forward(self, x):
        x = torch.tensor(x[0])
        l1 = F.relu(self.layer1(x))
        l2 = F.relu(self.layer2(l1))
        return self.layer3(l2)

    def act(self, state):
        pass

def make_env(env_id, seed) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    return env

def make_agent(env: gym.Env) -> Agent:
    agent = Agent(dim_actions=env.action_space.n, dim_obs=env.observation_space.shape[0])
    return agent

def make_things(env_name, seed) -> Tuple[gym.Env ,Agent]:
    env = make_env(env_name, seed)
    agent = make_agent(env)
    return env, agent

def run_things(env :gym.Env, agent: Agent):
    for _ in range(10):
        state = env.reset()
        while True:
            action = agent.forward(state)
            state = env.step(int(torch.argmax(action)))

def train_agent():
    env, agent = make_things("CartPole-v1", SEED)
    run_things(env, agent)

    return 5

train_agent()
if __name__ == "__main__":
    train_agent()
