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
learning_rate = 1e-5

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
        return Categorical(logits=self.policy_head(self.hidden(x))
            ).sample()

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
    return gym.make(env_id)

def make_agent(env: gym.Env) -> Agent:
    return Agent(dim_obs=env.observation_space.shape[0], dim_actions=env.action_space.n)

def make_env_agent(env_name, seed) -> Tuple[gym.Env ,Agent]:
    env = make_env(env_name, seed)
    agent = make_agent(env)
    return env, agent


def gae(rewards, values, gamma=0.99, lambda_=0.95):
    with torch.no_grad()
        advantages = []
        last_gae = 0.0
        last_value = values[-1]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * last_value - values[t]
            last_gae = delta + gamma * lambda_ * last_gae
            advantages.insert(0, last_gae)
            last_value = values[t]
        return np.array(advantages)

def rollout_agent_on_env(agent: Agent, env :gym.Env, eps=10):
    for n in range(eps):
        new_state = env.reset()
        old_state = new_state[0]
        while True:
            action = agent.forward(old_state)
            new_state, r, term, _, _ = env.step(int(torch.argmax(action)))
            mem.append((old_state, action, r, new_state))

            if term:
                break
            old_state = new_state

def train_agent(agent: Agent, env, updates):
    # Collect experience
    for _ in range(updates):
        rollout_agent_on_env(agent, env)
    print(mem[0])
    results = [[mem[i][2], agent.value(mem[i][0])] for i in range(len(mem))]
    rewards, values = zip(*results)
    print(gae(rewards, values))


    # Learn from experience
    """
    What do I need here? Lets just have a nice chat about it. I'm trying to
    impement PPO. I have sooome idea of what to do, but now I kind of got stuck.


    It looks like you're implementing a PPO (Proximal Policy Optimization) agent in
    PyTorch. To train the agent, you'll need to implement the following steps:

1.  **Compute advantages**: Calculate the advantage estimates using your experience buffer (`mem`). You can use the TD-error formula or another method.
2.  **Clipped surrogate objective**: Compute the clipped surrogate objective using the current policy and the old policy (the one from the previous iteration). This will help you update the policy while constraining its magnitude.
3.  **Policy update**: Update the agent's policy parameters based on the clipped surrogate objective.
4.  **Value function update**: Update the value function to better approximate the true values of states.
5.  **Entropy regularization**: Add an entropy term to encourage exploration.

We've had a productive conversation so far.

To summarize, we've discussed:

1.  **Computing advantages**: We explored the concept of computing advantages using
the TD-error formula and how it's used in reinforcement learning algorithms.
2.  **Generalized Advantage Estimation (GAE)**: We dove into GAE, which is a more
sophisticated method for estimating advantages. It uses a discounted sum of rewards,
weights future rewards by λ^t, and provides better estimates with reduced bias.
3.  **Why use GAE**: We talked about the benefits of using GAE, including its ability
to produce more accurate advantage estimates, reduce bias, and improve convergence.

Feel free to ask me anything or continue our conversation on any of these topics!

    I now think I comput the advantages. What comes next?

Now that you've computed the advantages, the next step is to:

1.  **Sample a batch from your memory**: Randomly select a subset (batch) of experiences from `mem`.
2.  **Compute the clipped surrogate objective**: This involves calculating the probability ratio between the new policy and the old policy for each action in the batch.
3.  **Clip the probability ratio**: Clip this ratio to prevent extreme updates to the policy.
4.  **Compute the final loss**: Use the clipped surrogate objective to compute the loss value for the batch.
5.  **Backpropagate and update the model parameters**: Use an optimizer (e.g., Adam) to update the policy network's parameters based on the loss.

This process will help you improve your policy by reducing the loss function, which measures how well your policy is at maximizing cumulative rewards.
    """
    pass

def run_agent():
    env, agent = make_env_agent("CartPole-v1", SEED)
    optimizer = torch.optim.Adam(agent.parameters(), learning_rate)
    train_agent(agent, env, updates=100)

    return 5

if __name__ == "__main__":
    print(run_agent())
