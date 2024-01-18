# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 20:23
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_reinforce.py.py
# @Software: CleanRL
# @Description: run_reinforce.py

import gym
import torch
from reinforce import REINFORCE, REINFORCEConfig


def init_layer(layers):
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.1)
            torch.nn.init.constant_(layer.bias, 0.)
        elif isinstance(layer, torch.nn.Sequential):
            init_layer(layer.children())


class PolicyNet(torch.nn.Module):
    def __init__(self, dim_observations, dim_actions, config, **kwargs):
        super(PolicyNet, self).__init__()
        self.config = config

        self.layer1 = torch.nn.Linear(dim_observations, 100)

        self.action_range = None
        if self.config.continuous_action:
            self.distribution = torch.distributions.Normal
            self.mean_layer = torch.nn.Linear(100, dim_actions)
            self.deviation_layer = torch.nn.Linear(100, dim_actions)
            self.action_range = kwargs['action_range']
        else:
            self.distribution = torch.distributions.Categorical
            self.layer2 = torch.nn.Linear(100, dim_actions)

        init_layer(self.children())

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        if self.config.continuous_action:
            mean = torch.tanh(self.mean_layer(x)) * self.action_range  # 取值[-action_range, action_range]
            deviation = torch.nn.functional.softplus(self.deviation_layer(x))  # 取值>0
            return mean, deviation
        else:
            probs = torch.softmax(self.layer2(x), dim=1)
            return probs


if __name__ == "__main__":
    config = REINFORCEConfig(epoches=10000, lr=1e-3, continuous_action=True, epoch_steps=200, gamma=0.98)
    ############### 创建环境 ###############
    env_name = 'Pendulum-v1' if config.continuous_action else 'CartPole-v1'
    env = gym.make(env_name, render_mode='rgb_array')
    env = env.unwrapped

    ############### 训练 ###############
    if config.continuous_action:
        policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0],
                               config, action_range=env.action_space.high[0])
    else:
        policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n,
                               config)
    rl = REINFORCE(env, policy_net, config)
    rl.learn()

    ############### 测试 ###############
    rl.env = gym.make(env_name, render_mode='human').unwrapped
    rl.play()
