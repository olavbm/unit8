import collections
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.nn import Linear, Sequential, Tanh
from wasabi import Printer

SEED = 1337
MEM_SIZE=2000
GAMMA = 0.999
msg = Printer()

mem = collections.deque(maxlen=MEM_SIZE)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, dim_obs, dim_actions):
        super().__init__()
        self.network = Sequential(
            layer_init(Linear(np.array(dim_obs).prod(), 64)),
            Tanh(),
            layer_init(Linear(64, 64)),
            Tanh(),
        )
        self.value_head = layer_init(Linear(64, 1), std=1.0)
        self.policy_head = layer_init(Linear(64, dim_actions), std=0.01)

    def hidden(self, x):
        x = torch.tensor(x)
        return self.network(x)

    def value(self, x):
        x = torch.tensor(x)
        return self.value_head(self.hidden(x))

    def action(self, x):
        x = torch.tensor(x)
        return Categorical(logits=self.policy_head(self.hidden(x))).sample()

    def forward(self, x):
        return self.action(x)

    # Maybe not needed
    """
    def td_estimate(self, state):
        return self.forward(state).max()

    # Maybe not needed
    @torch.no_grad()
    def td_target(self, state, reward, done):
        best_action = torch.argmax(self.forward(state))
        return (reward + (1-int(done) * (self.forward(state)[best_action])))
    """

def make_env(env_id, _) -> gym.Env:
    env = gym.make(env_id)
    return env

def make_agent(env: gym.Env) -> Agent:
    agent = Agent(dim_obs=env.observation_space.shape[0], dim_actions=env.action_space.n)
    return agent

def make_env_agent(env_name, seed) -> Tuple[gym.Env ,Agent]:
    env = make_env(env_name, seed)
    agent = make_agent(env)
    return env, agent

def train_agent_on_env(agent: Agent, env :gym.Env):
    for n in range(10):
        new_state = env.reset()
        old_state = new_state[0]
        while True:
            action = agent.forward(old_state)
            new_state, r, term, _, _ = env.step(int(torch.argmax(action)))
            mem.append((old_state, action, r, new_state))

            if term:
                break
            old_state = new_state

        if n % 3 == 0:
            print(agent.value(new_state))
            update_agent()

def update_agent():
    """
    1. replay mem
    2. loss func
    3. update net
    """
    pass

def run_agent():
    env, agent = make_env_agent("CartPole-v1", SEED)
    train_agent_on_env(agent, env)

    return 5

if __name__ == "__main__":
    print(run_agent())
