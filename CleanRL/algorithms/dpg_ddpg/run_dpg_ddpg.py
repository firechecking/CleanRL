# -*- coding: utf-8 -*-
# @Time    : 2024/2/7 11:51
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_dpg_ddpg.py
# @Software: CleanRL
# @Description: run_dpg_ddpg

import torch
import gym
from dpg_ddpg import DPGConfig, DPG


class ActorNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, action_range):
        super(ActorNet, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim, 32)
        self.layer2 = torch.nn.Linear(32, output_dim)

        self.action_range = action_range

    def forward(self, state):
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.tanh(x) * self.action_range
        return x


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()

        self.layer1 = torch.nn.Linear(state_dim, 32)
        self.layer2 = torch.nn.Linear(action_dim, 32)
        self.layer3 = torch.nn.Linear(32, 1)

    def forward(self, state, action):
        state = self.layer1(state)
        action = self.layer2(action)
        x = torch.nn.functional.relu(state + action)
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    env = gym.make('Pendulum-v1', render_mode='rgb_array').unwrapped

    actor = ActorNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    critic = CriticNet(env.observation_space.shape[0], env.action_space.shape[0])

    rl = DPG(env, actor, critic, DPGConfig())
    rl.learn()

    env = gym.make('Pendulum-v1', render_mode='human').unwrapped
    rl.env = env
    rl.play()
