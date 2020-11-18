try:
    from base_agent import Base_Agent
    from models import AtariDQN
    from env import construct_random_lane_env
except: pass

try:
    from .base_agent import Base_Agent
    from .models import AtariDQN
    from .env import construct_random_lane_env
except: pass

import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SAVE_PATH = "atari_dqn.pth.tar"
SAVE_LOOP = 100

# Hyperparameters --- don"t change, RL is very sensitive
learning_rate = 0.000001
gamma         = 0.98
buffer_limit  = 20000
batch_size    = 128
max_episodes  = 10000
t_max         = 600
min_buffer    = 10000
target_update = 10 # episode(s)
train_steps   = 10
max_epsilon   = 0.6
min_epsilon   = 0.0
epsilon_decay = 5000
print_interval= 100

Transition = collections.namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        """
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        """
        self.buffer = []
        self.buffer_limit = buffer_limit
        pass
    
    def push(self, transition):
        """
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        """
        self.buffer = (self.buffer + [transition])[-self.buffer_limit:] 
        pass
    
    def sample(self, batch_size):
        """
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        """
        sample = random.choices(self.buffer, k = batch_size)
        return (torch.stack([torch.tensor(s[0]).float().to(device) for s in sample]),
                torch.stack([torch.tensor(s[1]).long(). to(device) for s in sample]),
                torch.stack([torch.tensor(s[2]).float().to(device) for s in sample]),
                torch.stack([torch.tensor(s[3]).float().to(device) for s in sample]),
                torch.stack([torch.tensor(s[4]).float().to(device) for s in sample]))

    def __len__(self):
        """
        Return the length of the replay buffer.
        """
        return len(self.buffer)

def compute_loss(model, target, states, actions, rewards, next_states, dones):
    """
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    """
    model_values = model(states).gather(1, actions)
    
    target_values = target(next_states).max(dim = 1)[0].unsqueeze(1)
    target_values = rewards + gamma * target_values * (1 - dones)
    
    loss = F.smooth_l1_loss(model_values, target_values)
    return loss

def optimize(model, target, memory, optimizer):
    """
    Optimize the model for a sampled batch with a length of `batch_size`
    """
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    ### CHECK
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

def compute_epsilon(episode):
    """
    Compute epsilon used for epsilon-greedy exploration
    """
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

if __name__ == "__main__":
    env = construct_random_lane_env() 
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    target = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)

    if os.path.exists(SAVE_PATH):
        print("[DONE] Resuming training from previous training!")
        model .load_state_dict(torch.load(SAVE_PATH, map_location=lambda storage, loc: storage))
        target.load_state_dict(torch.load(SAVE_PATH, map_location=lambda storage, loc: storage))
    else:
        print("[DONE] Starting new training!")
    target.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print (model)

    agent  = Base_Agent()
    memory = ReplayBuffer()

    rewards = []
    losses = []
    print ("[Done] Initializing!")
    
    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)

        state = env.reset()
        agent.initialize()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            if np.random.random() < epsilon:
                action = agent.step(state)
            else:
                action = np.argmax(model(torch.stack([torch.Tensor(state).to(device)]))[0].detach().cpu().numpy())
                action = int(action)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break

        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) >= min_buffer:
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())
            target.eval()

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards), np.mean(losses), len(memory), epsilon*100))
            rewards = []
            losses = []

        if episode % SAVE_LOOP == 0:
            torch.save(model.state_dict(), SAVE_PATH)
