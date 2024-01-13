# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 21:46
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_train.py
# @Software: CleanRL
# @Description: run_train

import sys

sys.path.append('../../')

import numpy as np
import torch
from environment import WrappedEnv, init_env_args
from arguments import init_args

from CleanRL.common.layers import NoisyLinear
from dqn import DQN


class Net(torch.nn.Module):
    def __init__(self, dim_observation, n_actions, config):
        super(Net, self).__init__()
        self.dim_observation = dim_observation
        self.n_actions = n_actions
        self.config = config

        c, h, w = dim_observation

        ############ 卷积特征提取层 ############
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
        )

        ############ 动作价值/分布概率输出层 ############
        LinearLayer = NoisyLinear if self.config.noisy_net else torch.nn.Linear
        if self.config.dueling:
            self.value_layer = LinearLayer(512, self.config.distributional_atom_size)
            self.advantage_layer = LinearLayer(512, n_actions * self.config.distributional_atom_size)
        else:
            self.output_layer = LinearLayer(512, n_actions * self.config.distributional_atom_size)

        if self.config.distributional_atom_size > 1:
            support_z = torch.linspace(self.config.distributional_v_min, self.config.distributional_v_max, self.config.distributional_atom_size)
            self.register_buffer('support_z', support_z)

    def forward(self, batch_state, return_dist=False):
        x = batch_state
        x = self.feature_layer(x)
        if self.config.dueling:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            if self.config.distributional_atom_size <= 1:
                return value + advantage - advantage.mean()
            value = value.view(-1, 1, self.config.distributional_atom_size)
            advantage = advantage.view(-1, self.n_actions, self.config.distributional_atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
            dist = torch.softmax(q_atoms, dim=-1).clamp(min=1e-3)
        else:
            q = self.output_layer(x)
            if self.config.distributional_atom_size <= 1:
                return q
            q = q.view(-1, self.n_actions, self.config.distributional_atom_size)
            dist = torch.softmax(q, dim=-1).clamp(min=1e-3)

        if return_dist:
            return dist
        return torch.sum(dist * self.support_z, dim=2)

    def reset_noise(self):
        if not self.config.noisy_net: return
        if self.config.dueling:
            self.value_layer.reset_noise()
            self.advantage_layer.reset_noise()
        else:
            self.output_layer.reset_noise()

    def zero_noise(self):
        if not self.config.noisy_net: return
        if self.config.dueling:
            self.value_layer.w_epsilon.zero_()
            self.value_layer.b_epsilon.zero_()
            self.advantage_layer.w_epsilon.zero_()
            self.advantage_layer.b_epsilon.zero_()
        else:
            self.output_layer.w_epsilon.zero_()
            self.output_layer.b_epsilon.zero_()


def _print_args(config):
    print('############### config ###############')
    for k, v in config.__dict__.items():
        print('# ' + k + ':' + ' ' * (30 - len(k)) + str(v))
    print('############### config ###############')


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    ############### 初始化参数 ###############
    init_seed(999)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = init_args()
    parser = init_env_args(parser)
    args, _ = parser.parse_known_args()
    _print_args(args)

    ############### 创建环境 ###############
    stages = None
    if args.env_name.startswith('SuperMarioBrosRandomStages-'):  # TODO: 4-4, 8-4需要修改reward
        stages = ['1-1', '1-2', '1-3', '1-4',
                  '2-1', '2-2', '2-3', '2-4',
                  '3-1', '3-2', '3-3', '3-4',
                  '4-1', '4-2', '4-3',
                  '5-1', '5-2', '5-3', '5-4',
                  '6-1', '6-2', '6-3', '6-4',
                  '7-1', '7-2', '7-3',
                  '8-1', '8-2', '8-3', '8-4']
    env = WrappedEnv(args, stages=stages)
    state, info = env.reset()  # state.shape = (num_stack, resize, resize)

    ############### 模型训练 ###############
    q_net = Net(dim_observation=state.shape, n_actions=env.action_space.n, config=args)
    q_net = q_net.to(device)
    rl = DQN(env, q_net, args, device)
    rl.learn()

    ############### 模型测试 ###############
    step = 1000000
    rl.env = env
    for i in range(10):
        rl.play(ckpt_step=step, reset_noop=i, clear_noisy=False, reset_noisy=False, verbose=False, export_video_fn=f'noop_{i}.mp4')
