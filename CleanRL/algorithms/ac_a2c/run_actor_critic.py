# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 23:09
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_actor_critic.py
# @Software: CleanRL
# @Description: run_actor_critic

import torch
import gym
from actor_critic import ActorCritic, ActorCriticConfig


class ActorNet(torch.nn.Module):
    def __init__(self, dim_observations, dim_actions, action_range):
        super(ActorNet, self).__init__()
        self.action_range = action_range

        self.layer1 = torch.nn.Linear(dim_observations, 100)
        self.mean_layer = torch.nn.Linear(100, dim_actions)
        self.deviation_layer = torch.nn.Linear(100, dim_actions)

        self.distribution = torch.distributions.Normal

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        mean = torch.tanh(self.mean_layer(x)) * self.action_range  # 取值[-action_range, action_range]
        deviation = torch.nn.functional.softplus(self.deviation_layer(x)) + 0.001  # 取值>0
        return mean, deviation


class CriticNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim,32)
        self.layer2 = torch.nn.Linear(32, 1)

    def forward(self, state):
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    ############### 创建环境 ###############
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = env.unwrapped

    ############### 训练 ###############
    actor = ActorNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    critic = CriticNet(env.observation_space.shape[0])
    rl = ActorCritic(env, actor, critic, ActorCriticConfig())
    rl.learn()

    ############### 测试 ###############
    env = gym.make('Pendulum-v1', render_mode='human').unwrapped
    rl.env = env
    rl.play()
