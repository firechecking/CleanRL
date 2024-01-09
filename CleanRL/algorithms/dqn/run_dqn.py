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
from CleanRL.common.layers import NoisyLinear


class QNet(torch.nn.Module):
    def __init__(self, n_observations, n_actions, config):
        super(QNet, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.config = config

        LinearLayer = NoisyLinear if self.config.noisy else torch.nn.Linear
        
        self.layer1 = torch.nn.Linear(n_observations, 320)
        if self.config.dueling:
            self.layer_v = LinearLayer(320, 1)
            self.layer_a = LinearLayer(320, n_actions)
        else:
            self.layer_q = LinearLayer(320, n_actions)

    def forward(self, batch_state):
        x = torch.relu(self.layer1(batch_state))
        if self.config.dueling:
            v = self.layer_v(x)
            a = self.layer_a(x)
            return v + a - a.mean()
        else:
            return self.layer_q(x)

    def reset_noise(self):
        if not self.config.noisy: return
        if self.config.dueling:
            self.layer_v.reset_noise()
            self.layer_a.reset_noise()
        else:
            self.layer_q.reset_noise()

    def zero_noise(self):
        ############### 将w_epsilon、b_epsilon都置0，清除noisy (play阶段可以尝试) ###############
        if not self.config.noisy: return
        if self.config.dueling:
            self.layer_v.w_epsilon.zero_()
            self.layer_v.b_epsilon.zero_()
            self.layer_a.w_epsilon.zero_()
            self.layer_a.b_epsilon.zero_()
        else:
            self.layer_q.w_epsilon.zero_()
            self.layer_q.b_epsilon.zero_()


def _print_args(config):
    print('############### config ###############')
    for k, v in config.__dict__.items():
        print('# ' + k + ':' + ' ' * (30 - len(k)) + str(v))
    print('############### config ###############')


if __name__ == "__main__":
    ############### 创建环境 ###############
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
    config = DQNConfig()
    _print_args(config)
    q_net = QNet(env.observation_space.shape[0], env.action_space.n, config=config)

    ############### 训练 ###############
    rl = DQN(env, q_net, config)
    # rl.learn()

    ############### 测试 ###############
    rl.env = gym.make('CartPole-v1', render_mode='human').unwrapped
    rl.play(epoch=599)
