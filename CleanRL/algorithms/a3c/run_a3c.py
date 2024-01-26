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


def init_layer(layers):
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.1)
            torch.nn.init.constant_(layer.bias, 0.)
        elif isinstance(layer, torch.nn.Sequential):
            init_layer(layer.children())


class ActorCriticNet(torch.nn.Module):
    def __init__(self, input_dim, dim_actions, action_range, model_type='combine'):
        super(ActorCriticNet, self).__init__()
        self.action_range = action_range
        self.model_type = model_type

        self.shared_layer = torch.nn.Linear(input_dim, 32)

        if model_type != 'critic':
            ############### 包含actor相关网络 ###############
            self.mean_layer = torch.nn.Linear(32, dim_actions)
            self.deviation_layer = torch.nn.Linear(32, dim_actions)
            self.distribution = torch.distributions.Normal

        if model_type != 'actor':
            ############### 包含critic相关网络 ###############
            self.value_layer = torch.nn.Linear(32, 1)

        init_layer(self.children())

    def forward(self, state):
        hidden = torch.relu(self.shared_layer(state))
        mean, deviation, value = None, None, None

        if self.model_type != 'critic':
            mean = torch.tanh(self.mean_layer(hidden)) * self.action_range  # 取值[-action_range, action_range]
            deviation = torch.nn.functional.softplus(self.deviation_layer(hidden)) + 0.001  # 取值>0
        if self.model_type != 'actor':
            value = self.value_layer(hidden)

        if self.model_type == 'actor':
            return mean, deviation
        elif self.model_type == 'critic':
            return value
        else:
            return mean, deviation, value


def env_and_net_builder(render_model='rgb_array', unique_net=False):
    env = gym.make('Pendulum-v1', render_mode=render_model)  # https://gymnasium.farama.org/environments/classic_control/pendulum/
    env = env.unwrapped
    if unique_net:
        ############### 分别定义actor、critic ###############
        actor = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], model_type='actor')
        critic = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], model_type='critic')
        return env, [actor, critic]
    else:
        ############### 定义合并的actor-critic ###############
        actor_critic = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], model_type='combine')
        return env, [actor_critic, ]


if __name__ == "__main__":
    rl = A3C(env_and_net_builder, A3CConfig())
    rl.learn()
    rl.play()
