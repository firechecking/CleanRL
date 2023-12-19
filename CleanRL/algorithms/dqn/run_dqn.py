# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 23:19
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_dqn.py
# @Software: CleanRL
# @Description: run_dqn

import torch
import gym
from dqn import DQN, DQNConfig


class QNet(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.layer1 = torch.nn.Linear(n_observations, 100)
        self.layer2 = torch.nn.Linear(100, n_actions)

    def forward(self, batch_state):
        x = torch.relu(self.layer1(batch_state))
        return self.layer2(x)


if __name__ == "__main__":
    ############### 创建环境 ###############
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
    q_net = QNet(env.observation_space.shape[0], env.action_space.n)

    ############### 训练 ###############
    rl = DQN(env, q_net, DQNConfig())
    rl.learn()

    ############### 测试 ###############
    rl.env = gym.make('CartPole-v1', render_mode='human').unwrapped
    rl.play(epoch=799)
