# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 20:18
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : human_play.py
# @Software: CleanRL
# @Description: human_play

import time
import numpy as np
from pyglet.window import key
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as MOVEMENT

from CleanRL.common.video_writer import PreviewWindow
from environment import WrappedEnv, init_env_args
from arguments import init_args


class ControllerWindow(PreviewWindow):
    ############### 用于监听键盘事件、显示画面 ###############
    def __init__(self, *args, **kwargs):
        super(ControllerWindow, self).__init__(*args, **kwargs)
        self._pressed_keys = []

    def on_key_press(self, symbol, modifiers):
        self._pressed_keys.append(symbol)

    def on_key_release(self, symbol, modifiers):
        self._pressed_keys.remove(symbol)


def match_key(keys):
    ############### 将键盘事件映射到游戏action ###############
    move_map = {'right': key.D, 'left': key.A, 'down': key.S, 'up': key.W,
                'A': key.K, 'B': key.J,
                'NOOP': 0}

    matched_movement = []
    for move in MOVEMENT:
        matched = [1 if move_map[k] in keys else 0 for k in move]
        if sum(matched) == len(move) and len(move) > len(matched_movement):
            matched_movement = move

    return 0 if len(matched_movement) == 0 else MOVEMENT.index(matched_movement)


def human_play(env_name):
    total_reward = 0
    step = 0
    ############### 初始化环境 ###############
    wrapped_env = True
    if wrapped_env:
        parser = init_args()
        parser = init_env_args(parser)
        args, _ = parser.parse_known_args()
        args.env_name = 'SuperMarioBros-1-1-v0'
        args.optimize_reward = True
        args.resize = 0
        args.gray_scale = False
        env = WrappedEnv(args)
    else:
        env = gym_super_mario_bros.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
        env = JoypadSpace(env, MOVEMENT)

    state, _ = env.reset()
    controller_window = ControllerWindow()

    ############### 游戏主循环 ###############
    while True:
        if wrapped_env:
            if args.gray_scale:
                state = state[-1].squeeze(0)
                state = np.repeat(state[:, :, np.newaxis], 3, axis=2).numpy()
            else:
                state = np.transpose(state[-1, :, :, :].numpy(), (1, 2, 0))
            if args.resize:
                state = state * 255
            state = state.astype(np.uint8)

        controller_window.show(state)
        step += 1

        ############### 选择action ###############
        action = 0
        if len(controller_window._pressed_keys) > 0:
            action = match_key(controller_window._pressed_keys)

        ############### 执行action ###############
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward

        print('step: {}, action: {}, reward: {}, total_reward: {}, x_pos: {}'.format(step, action, reward, total_reward, info['x_pos']))
        if done:
            time.sleep(1)
            break

        if wrapped_env:
            time.sleep(1 / 15)
        else:
            time.sleep(1 / 60)


if __name__ == "__main__":
    human_play('SuperMarioBros-v0')
