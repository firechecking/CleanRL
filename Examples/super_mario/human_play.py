# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 20:18
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : human_play.py
# @Software: CleanRL
# @Description: human_play

import time, pyglet
from pyglet.window import key
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as MOVEMENT


class MyWindow(pyglet.window.Window):
    ############### 主窗体，用于监听键盘事件、显示画面 ###############
    def __init__(self, *args, **kwargs):
        super().__init__(width=420, height=420, vsync=False, resizable=True, *args, **kwargs)
        self._pressed_keys = []

    def on_key_press(self, symbol, modifiers):
        self._pressed_keys.append(symbol)

    def on_key_release(self, symbol, modifiers):
        self._pressed_keys.remove(symbol)

    def show(self, frame):
        self.clear()
        self.switch_to()
        self.dispatch_events()

        image = pyglet.image.ImageData(
            frame.shape[1],
            frame.shape[0],
            'RGB',
            frame.tobytes(),
            pitch=frame.shape[1] * -3
        )
        image.blit(0, 0, width=self.width, height=self.height)
        self.flip()


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
    env = gym_super_mario_bros.make(env_name, apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, MOVEMENT)
    state, _ = env.reset()
    controller_window = MyWindow()

    ############### 游戏主循环 ###############
    while True:
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

        time.sleep(1 / 60)


if __name__ == "__main__":
    human_play('SuperMarioBros-v0')
