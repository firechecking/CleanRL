# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 18:36
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_ppo.py
# @Software: CleanRL
# @Description: run_ppo.py

import torch
import gym
from ppo import PPO, PPOConfig


class ActorNet(torch.nn.Module):
    def __init__(self, dim_observations, dim_actions, action_range):
        super(ActorNet, self).__init__()
        self.action_range = action_range

        self.layer1 = torch.nn.Linear(dim_observations, 128)
        self.mean_layer = torch.nn.Linear(128, dim_actions)
        self.deviation_layer = torch.nn.Linear(128, dim_actions)

        self.distribution = torch.distributions.Normal

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        mean = torch.tanh(self.mean_layer(x)) * self.action_range  # 取值[-action_range, action_range]
        deviation = torch.nn.functional.softplus(self.deviation_layer(x)) + 0.001  # 取值>0
        return mean, deviation


class CriticNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim, 128)
        self.layer2 = torch.nn.Linear(128, 1)

    def forward(self, state):
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        return self.layer2(x)


if __name__ == "__main__":
    ############### 初始化 ###############
    torch.manual_seed(999)

    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = env.unwrapped

    actor = ActorNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    critic = CriticNet(env.observation_space.shape[0])

    ############### 训练 ###############
    rl = PPO(env, actor, critic, PPOConfig())
    rl.learn()

    ############### 测试 ###############
    env = gym.make('Pendulum-v1', render_mode='human').unwrapped
    rl.env = env
    rl.play()
