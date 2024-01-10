# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 19:07
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : environment.py
# @Software: CleanRL
# @Description: environment

import random
from collections import deque

import torch
import numpy as np
from torchvision import transforms as T

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


def init_env_args(parser):
    group = parser.add_argument_group(title='interactive arguments')
    group.add_argument('--env_name', default='SuperMarioBros-v0', type=str)
    group.add_argument('--actions', default='complex', choices=('simple', 'complex'))
    group.add_argument('--skip_frames', default=4, type=int)
    group.add_argument('--resize', default=84, type=int)
    group.add_argument('--num_stack', default=4, type=int)
    group.add_argument('--no_gray_scale', dest='gray_scale', default=True, action='store_false')
    group.add_argument('--optimize_reward', default=False, action='store_true')
    return parser


class WrappedEnv(JoypadSpace):
    def __init__(self, config, stages=None, render_mode='rgb_array'):
        if stages is not None:
            env = gym_super_mario_bros.make(config.env_name, apply_api_compatibility=True, render_mode=render_mode, stages=stages)
        else:
            env = gym_super_mario_bros.make(config.env_name, apply_api_compatibility=True, render_mode=render_mode)
        super(WrappedEnv, self).__init__(env, SIMPLE_MOVEMENT if config.actions == 'simple' else COMPLEX_MOVEMENT)

        self.config = config

        self.stack_frames = deque(maxlen=self.config.num_stack)
        self.last_info = None

    def step(self, action):
        total_reward = 0
        ############ 跳过帧 ############
        skip_frames = max(self.config.skip_frames, 1)
        for i in range(skip_frames):
            observation, reward, terminated, truncated, info = super(WrappedEnv, self).step(action)
            total_reward += self._optimize_reward(reward, terminated or truncated, info)
            if terminated or truncated:
                break

        return self._process_one_observation(observation), total_reward, terminated, truncated, info

    def reset(self):
        observation, info = super(WrappedEnv, self).reset()
        ############ 初始化时随机跳过帧，增加随机性 ############
        for i in range(random.randint(0, 100)):
            observation, _, _, _, _ = super(WrappedEnv, self).step(0)
        return self._process_one_observation(observation, reset=True), info

    def _process_one_observation(self, observation, reset=False):
        observation = np.transpose(observation, (2, 0, 1))  # [H, W, C] to [C, H, W]
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        ############ 转灰度图 ############
        if self.config.gray_scale:
            transform = T.Grayscale()
            observation = transform(observation)  # shape= (1,240,256)

        ############ 尺寸resize ############
        if self.config.resize > 0:
            transforms = T.Compose(
                [T.Resize((self.config.resize, self.config.resize), antialias=True), T.Normalize(0, 255)]
            )
            observation = transforms(observation).squeeze(0)  # shape = (self.config.resize, self.config.resize)

        ############ 连续多帧组合 ############
        if reset:  # 初始化时，同一个画面重复num_stack次
            for _ in range(self.config.num_stack):
                self.stack_frames.append(observation)
        else:
            self.stack_frames.append(observation)

        return torch.stack([frame for frame in self.stack_frames])

    def _optimize_reward(self, reward, done, info):
        if not self.config.optimize_reward: return reward

        reward_range = (-15, 15)

        if done:
            self.last_info = None
            if info['flag_get']:  # 通关
                return reward_range[1]
            return reward

        if self.last_info is not None and reward != self.unwrapped.env.reward_range[1]:
            all_status = ('small', 'tall', 'fireball')

            if info['flag_get']:  # 通关
                reward = reward_range[1]
            elif all_status.index(info['status']) - all_status.index(self.last_info['status']) != 0:  # 变身
                reward = 2 * all_status.index(info['status']) - all_status.index(self.last_info['status'])
            elif info['coins'] > self.last_info['coins']:  # 金币
                reward = 2
            elif info['score'] > self.last_info['score']:  # 分数
                reward = 2
            elif info['life'] > self.last_info['life']:  # 增加生命
                reward = 6
            elif info['life'] < self.last_info['life']:  # 减少生命
                reward = -6
            elif info['x_pos'] - self.last_info['x_pos'] > 5 or info['x_pos'] - self.last_info['x_pos'] < -5:  # 重新开始
                reward = 0
            elif info['x_pos'] > self.last_info['x_pos']:  # 前进
                reward = 1
            elif (info['x_pos'] == self.last_info['x_pos']):
                reward = -1
            else:
                reward = -1

        self.last_info = info
        return max(min(reward, reward_range[1]), reward_range[0])


if __name__ == "__main__":
    pass
