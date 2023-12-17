# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 16:20
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_q_learning.py
# @Software: CleanRL
# @Description: run_q_learning

import gym
from q_learning import QLearning, QLearningConfig

if __name__ == "__main__":
    ############### 创建环境 ###############
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    ############### 训练 ###############
    rl = QLearning(env, QLearningConfig())
    rl.learn()

    ############### 测试 ###############
    rl.env = gym.make('CartPole-v1', render_mode='human').unwrapped
    rl.play()
