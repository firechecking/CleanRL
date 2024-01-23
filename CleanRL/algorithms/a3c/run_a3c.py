# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 19:56
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_a3c.py
# @Software: CleanRL
# @Description: run_a3c

import torch
import gym
from a3c import A3CConfig, A3C


class ActorCriticNet(torch.nn.Module):
    def __init__(self, input_dim, dim_actions, action_range):
        super(ActorCriticNet, self).__init__()
        self.action_range = action_range

        self.shared_layer = torch.nn.Linear(input_dim, 100)

        self.mean_layer = torch.nn.Linear(100, dim_actions)
        self.deviation_layer = torch.nn.Linear(100, dim_actions)
        self.distribution = torch.distributions.Normal

        self.value_layer = torch.nn.Linear(100, 1)

    def forward(self, state):
        hidden = torch.relu(self.shared_layer(state))

        mean = torch.tanh(self.mean_layer(hidden)) * self.action_range  # 取值[-action_range, action_range]
        deviation = torch.nn.functional.softplus(self.deviation_layer(hidden)) + 0.001  # 取值>0
        value = self.value_layer(hidden)

        return mean, deviation, value


def env_and_net_builder(render_model='rgb_array'):
    env = gym.make('Pendulum-v1', render_mode=render_model)  # https://gymnasium.farama.org/environments/classic_control/pendulum/
    env = env.unwrapped
    net = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    return env, net


if __name__ == "__main__":
    rl = A3C(env_and_net_builder, A3CConfig())
    rl.learn()
    rl.play()
