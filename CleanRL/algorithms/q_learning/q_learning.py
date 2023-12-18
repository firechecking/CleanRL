# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 16:20
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : q_learning.py
# @Software: CleanRL
# @Description: q_learning

import os, random, pickle, time
import numpy as np


class QLearningConfig():
    def __init__(self, **kwargs):
        self.epoches = 10000
        self.epoch_steps = 500
        self.e_greedy_start = 0.95
        self.e_greedy_end = 0.1
        self.e_greedy_decay = 2000
        self.gamma = 0.9
        self.lr = 0.1
        self.load_path = 'q_table.pkl'
        self.save_interval = 100

        for k, v in kwargs.items():
            setattr(self, k, v)


class QLearning():
    def __init__(self, env, config: QLearningConfig):
        self.env = env
        self.config = config

        self.q_table = pickle.load(open(self.config.load_path, 'rb')) if os.path.exists(self.config.load_path) else {}

        self.e_greedy = self.config.e_greedy_start
        self.e_greedy_decay_per_epoch = (self.config.e_greedy_end - self.config.e_greedy_start) / self.config.e_greedy_decay

    def _simple_maybe_add_state(self, state):
        '''
        q_learning中要创建1个(state、action、value)的关系表，如果state数量太大，每个state被更新的次数会过少，所以这里对state进行了简化
        '''
        state = state * 4
        state = state.astype(int)

        if str(state) not in self.q_table:
            self.q_table[str(state)] = np.array([0.] * self.env.action_space.n)
        return state

    def learn(self):
        for epoch in range(self.config.epoches):
            ############### 重置环境 ###############
            state, _ = self.env.reset()
            state = self._simple_maybe_add_state(state)

            epoch_reward = 0
            for epoch_step in range(self.config.epoch_steps):
                ############### 选择action ###############
                if random.random() > self.e_greedy:
                    action = np.argmax(self.q_table[str(state)])
                else:
                    action = self.env.action_space.sample()

                ############### 执行action ###############
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._simple_maybe_add_state(next_state)
                done = terminated or truncated
                if done:
                    reward = -10
                epoch_reward += reward

                ############### 更新q表 ###############
                q_observation = reward + self.config.gamma * np.max(self.q_table[str(next_state)])
                self.q_table[str(state)][action] += self.config.lr * (q_observation - self.q_table[str(state)][action])

                if done or epoch_step >= self.config.epoch_steps - 1:
                    print('epoch: {}, steps: {}, greedy: {}, reward: {}'.format(epoch, epoch_step, self.e_greedy, epoch_reward))
                    ############### 保存q表到文件 ###############
                    if (epoch + 1) % self.config.save_interval == 0:
                        pickle.dump(self.q_table, open(self.config.load_path, 'wb'))
                    break

                state = next_state
            self.e_greedy = max(self.config.e_greedy_end, self.e_greedy + self.e_greedy_decay_per_epoch)

    def play(self):
        print('start play...')
        ############### 加载训练后的q表 ###############
        self.q_table = pickle.load(open(self.config.load_path, 'rb'))

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            state = self._simple_maybe_add_state(state)
            action = np.argmax(self.q_table[str(state)])

            ############### 执行action ###############
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
            if done:
                time.sleep(2)
                break


if __name__ == "__main__":
    pass
