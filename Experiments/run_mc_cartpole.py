# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 20:45
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : run_mc_cartpole.py
# @Software: CleanRL
# @Description: run_mc_cartpole

import pickle, time
from collections import defaultdict
from tqdm import trange
import gym


def simple_state(state):
    state = state * 4
    state = state.astype(int)
    return state


class MC_CartpoleConfig():
    def __init__(self):
        self.epoches = 50000
        self.gamma = 0.9
        self.save_ = 100
        self.e_greedy_start = 0.95
        self.load_path = 'mc.pkl'


class MC_Cartpole():
    def __init__(self, env, config: MC_CartpoleConfig):
        self.env = env
        self.config = config

    def learn(self):
        print('start train...')
        mc_values_list = defaultdict(list)
        for epoch in trange(self.config.epoches):
            state, _ = self.env.reset()
            state = simple_state(state)
            replay_buffer = []  # 完整记录局内state、action路径
            for epoch_step in range(500):
                ############### MC训练阶段都是随机选择动作 ###############
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = simple_state(next_state)
                done = terminated or truncated
                if done:
                    reward = -10
                replay_buffer.append((state, action, reward))

                if done:
                    ############### 一局结束，按局内路径从后往前计算价值 ###############
                    discounted_reward = 0
                    for state, action, reward in reversed(replay_buffer):
                        discounted_reward = reward + self.config.gamma * discounted_reward
                        mc_values_list[(str(state), action)].append(discounted_reward)
                    break

                state = next_state

        ############### 统计平均 ###############
        mc_values = {}
        for k, v in mc_values_list.items():
            mc_values[k] = sum(v) / len(v)
        pickle.dump(mc_values, open(self.config.load_path, 'wb'))

    def play(self):
        print('start play...')
        ############### 加载平均价值 ###############
        mc_values = pickle.load(open(self.config.load_path, 'rb'))

        total_reward, step = 0, 0
        state, _ = self.env.reset()
        while True:
            step += 1
            state = simple_state(state)
            ############### 选择action ###############
            max_value, action = 0, 0
            for _a in range(self.env.action_space.n):
                if (str(state), _a) in mc_values:
                    if mc_values[(str(state), _a)] > max_value:
                        max_value, action = mc_values[(str(state), _a)], _a

            ############### 执行action ###############
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
            if done:
                time.sleep(2)
                break


if __name__ == "__main__":
    ############### 创建环境 ###############
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    ############### 训练 ###############
    rl = MC_Cartpole(env, MC_CartpoleConfig())
    rl.learn()

    ############### 测试 ###############
    rl.env = gym.make('CartPole-v1', render_mode='human').unwrapped
    rl.play()
