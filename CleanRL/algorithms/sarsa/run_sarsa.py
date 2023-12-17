# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 21:17
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_sarsa.py
# @Software: CleanRL
# @Description: run_sarsa


import gym
from sarsa import Sarsa, SarsaConfig

if __name__ == "__main__":
    ############### 创建环境 ###############
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    ############### 训练 ###############
    rl = Sarsa(env, SarsaConfig())
    # rl.learn()

    ############### 测试 ###############
    rl.env = gym.make('CartPole-v1', render_mode='human').unwrapped
    rl.play()
