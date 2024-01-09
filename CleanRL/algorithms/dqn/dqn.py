# -*- coding: utf-8 -*-
# @Time    : 2023/12/17 23:19
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : dqn.py
# @Software: CleanRL
# @Description: dqn

import os, random, time, copy
import numpy as np
import torch
from CleanRL.common.sum_tree import SumTree


def stack_data(batch_data, device='cpu'):
    ############### 将list数据转成tensor, shape=(batch_size, n_observations) ###############
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data).to(device)
    else:
        return torch.tensor(batch_data).to(device)


class DQNConfig():
    def __init__(self, **kwargs):
        self.epoches = 1000
        self.epoch_steps = 500
        self.e_greedy_start = 0.95
        self.e_greedy_end = 0.1
        self.e_greedy_decay = 500
        self.gamma = 0.9
        self.lr = 1e-4
        self.save_interval = 100
        self.load_path = None
        self.save_ckpt_path = './q_net/'

        self.replay_buffer_size = 10000
        self.replay_warm_up = 1000
        self.batch_size = 32
        self.tau = 0.001

        self.double_q = True
        self.prioritized_replay = True
        self.prioritized_alpha = 0.6
        self.prioritized_beta = 0.4

        self.n_step_learning = 3

        for k, v in kwargs.items():
            setattr(self, k, v)


class DQN():
    def __init__(self, env, q_net, config: DQNConfig, device='cpu'):
        self.env = env
        self.q_net = q_net
        self.config = config
        self.device = device

        if self.config.load_path and os.path.exists(self.config.load_path):
            self.q_net.load_state_dict(torch.load(self.config.load_path, map_location=device))

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)

        self.e_greedy = self.config.e_greedy_start
        self.e_greedy_decay_per_epoch = (self.config.e_greedy_end - self.config.e_greedy_start) / self.config.e_greedy_decay

        os.makedirs(self.config.save_ckpt_path, exist_ok=True)

        self.t_net = copy.deepcopy(self.q_net)
        self.t_net.eval()
        self.replay_buffer = []
        if self.config.prioritized_replay:
            self.replay_buffer = SumTree(self.config.replay_buffer_size)

    def learn(self):
        print('start train...')
        self.q_net.train()
        for epoch in range(self.config.epoches):
            ############### 重置环境 ###############
            state, _ = self.env.reset()
            epoch_reward = 0
            for epoch_step in range(self.config.epoch_steps):
                ############### 选择action ###############
                if random.random() > self.e_greedy:
                    with torch.no_grad():
                        batch_state = stack_data([state, ], self.device)
                        q_values = self.q_net(batch_state)
                        action = torch.argmax(q_values).cpu().item()
                else:
                    action = self.env.action_space.sample()

                ############### 执行action ###############
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if done:
                    reward = -10
                epoch_reward += reward

                ############### replay buffer ###############
                if self.config.prioritized_replay:
                    self.replay_buffer.add(data=(state, action, reward, next_state, done), priority=None)
                else:
                    self.replay_buffer.append((state, action, reward, next_state, done))
                    if len(self.replay_buffer) > self.config.replay_buffer_size:
                        self.replay_buffer.pop(0)

                if len(self.replay_buffer) >= self.config.replay_warm_up:
                    self._one_batch_train()
                    ############### 增量式更新t_net模型 ###############
                    for param, target_param in zip(self.q_net.parameters(), self.t_net.parameters()):
                        target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

                if done or epoch_step >= self.config.epoch_steps - 1:
                    print('epoch: {}, steps: {}, greedy: {}, reward: {}'.format(epoch, epoch_step, self.e_greedy, epoch_reward))
                    ############### 保存模型 ###############
                    if (epoch + 1) % self.config.save_interval == 0:
                        torch.save(self.q_net.state_dict(),
                                   os.path.join(self.config.save_ckpt_path, 'ckpt_{}.pth'.format(epoch)))
                    break

                state = next_state
            self.e_greedy = max(self.config.e_greedy_end, self.e_greedy + self.e_greedy_decay_per_epoch)

    def _sample_batch_data(self):
        ############ 采样batch_size组n-step数据 ############
        bias_weight = None
        if self.config.prioritized_replay:
            batch_idx, batch_seq_data, sampling_weight = self.replay_buffer.sample(self.config.batch_size, self.config.n_step_learning)
            ############ 计算重要性采样补偿系数 ############
            bias_max_weight = (len(self.replay_buffer) * self.replay_buffer.min_priority) ** (-self.config.prioritized_beta)
            bias_weight = (len(self.replay_buffer) * np.array(sampling_weight)) ** (-self.config.prioritized_beta)
            bias_weight = torch.tensor(bias_weight / bias_max_weight, dtype=torch.float32, device=self.device)
        else:
            batch_idx = random.sample(range(len(self.replay_buffer)), self.config.batch_size)
            batch_seq_data = []
            for idx in batch_idx:
                seq_data = [self.replay_buffer[idx]]
                while len(seq_data) < self.config.n_step_learning:
                    if seq_data[-1][4]: break
                    if idx + len(seq_data) >= len(self.replay_buffer): break
                    seq_data.append(self.replay_buffer[idx + len(seq_data)])
                batch_seq_data.append(seq_data)

        ############ 用累积的n-step数据重构batch数据 ############
        batch_data = []
        for seq_data in batch_seq_data:
            data = list(seq_data.pop(0))  # state, action, reward, next_state, done
            data.append(len(seq_data) + 1)  # 记录每个训练样本的step数
            if len(seq_data) > 0:  # 使用第n步next_state代替第1步next_state
                data[3] = seq_data[-1][3]
            ############### 按n-step公式计算reward ###############
            for i, _data in enumerate(seq_data):
                data[2] += (self.config.gamma ** (i + 1)) * _data[2]
            batch_data.append(data)

        batch_state = stack_data([data[0] for data in batch_data], self.device)
        batch_action = stack_data([data[1] for data in batch_data], self.device)
        batch_reward = stack_data([data[2] for data in batch_data], self.device)
        batch_next_state = stack_data([data[3] for data in batch_data], self.device)
        batch_n_step = [data[5] for data in batch_data]
        return batch_state, batch_action, batch_reward, batch_next_state, batch_n_step, batch_idx, bias_weight

    def _one_batch_train(self):
        ############ 构造batch数据 ############
        state, action, reward, next_state, n_step, batch_idx, bias_weight = self._sample_batch_data()

        ############### 计算loss ###############
        with torch.no_grad():
            ############### 按n-step公式计算gamma ###############
            gamma = stack_data([self.config.gamma ** n for n in n_step], self.device)
            if self.config.double_q:
                t_action = torch.argmax(self.q_net(next_state), dim=-1)
            else:
                t_action = torch.argmax(self.t_net(next_state), dim=-1)
            q_observation = reward + gamma * self.t_net(next_state).gather(1, t_action.unsqueeze(1)).squeeze(1)

        ############### 计算loss ###############
        q_eval = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        criterion = torch.nn.SmoothL1Loss(reduction='none')
        loss = criterion(q_eval, q_observation)

        if self.config.prioritized_replay:
            ############ 更新priority ############
            for idx, priority in zip(batch_idx, loss):
                self.replay_buffer.update_priority(idx, torch.pow(priority.cpu().detach() + 1e-2, self.config.prioritized_alpha).item())

            ############ 重要性采样补偿 ############
            loss = torch.mul(loss, bias_weight)

        ############### 更新模型 ###############
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def play(self, epoch):
        print('start play...')
        ############### 加载训练后的模型 ###############
        ckpt_fn = os.path.join(self.config.save_ckpt_path, 'ckpt_{}.pth'.format(epoch))
        self.q_net.load_state_dict(torch.load(ckpt_fn, map_location='cpu'))
        self.q_net.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            q_values = self.q_net(stack_data([state, ]))
            action = torch.argmax(q_values).cpu().item()  # 模型最优动作

            ############### 执行action ###############
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            is_done = terminated or truncated

            print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
            if is_done:
                time.sleep(2)
                break


if __name__ == "__main__":
    pass
