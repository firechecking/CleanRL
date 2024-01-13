# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 21:37
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : dqn.py
# @Software: CleanRL
# @Description: dqn

import os, random, time, copy, json
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


class DQN():
    def __init__(self, env, q_net, config, device='cpu'):
        self.env = env
        self.q_net = q_net
        self.config = config
        self.device = device

        if self.config.load_path and os.path.exists(self.config.load_path):
            self.q_net.load_state_dict(torch.load(self.config.load_path, map_location=device))

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr, eps=1.5e-4)

        self.e_greedy = self.config.e_greedy_start
        self.e_greedy_decay_per_epoch = (self.config.e_greedy_end - self.config.e_greedy_start) / self.config.e_greedy_decay

        os.makedirs(self.config.save_ckpt_path, exist_ok=True)
        json.dump(self.config.__dict__,
                  open(os.path.join(self.config.save_ckpt_path, 'config.json'), 'w'),
                  ensure_ascii=False, indent=4)

        self.t_net = copy.deepcopy(self.q_net)
        self.t_net.eval()
        self.replay_buffer = []
        if self.config.prioritized_replay:
            self.replay_buffer = SumTree(self.config.replay_buffer_size)
            self.prioritized_beta = self.config.prioritized_beta

    def learn(self):
        print('start train...')
        self.q_net.train()

        epoch, trained_step = 0, 0
        for frame in range(1, self.config.frames + 1):
            ############### 重置环境 ###############
            epoch += 1
            state, _ = self.env.reset()
            epoch_reward = 0

            ############### 增大优先采样权重 ###############
            fraction = min(frame / self.config.frames, 1.0)
            self.prioritized_beta = self.prioritized_beta + fraction * (1.0 - self.prioritized_beta)

            for epoch_step in range(self.config.epoch_steps):
                ############### 每一步都随机生成新的噪音 ###############
                self.q_net.reset_noise()
                self.t_net.reset_noise()

                ############### 选择action ###############
                if self.config.noisy_net or random.random() > self.e_greedy:
                    with torch.no_grad():
                        batch_state = stack_data([state, ], self.device)
                        q_values = self.q_net(batch_state)
                        action = torch.argmax(q_values).cpu().item()
                else:
                    action = self.env.action_space.sample()

                ############### 执行action ###############
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                # if done:
                #     reward = -10
                epoch_reward += reward

                ############### replay buffer ###############
                if self.config.prioritized_replay:
                    self.replay_buffer.add(data=(state, action, reward, next_state, done), priority=None)
                else:
                    self.replay_buffer.append((state, action, reward, next_state, done))
                    if len(self.replay_buffer) > self.config.replay_buffer_size:
                        self.replay_buffer.pop(0)

                loss = None
                if len(self.replay_buffer) >= min(self.config.replay_buffer_size, self.config.replay_warm_up):
                    trained_step += 1
                    loss = self._one_batch_train()
                    if self.config.tau < 1:
                        ############### 增量式更新t_net模型 ###############
                        for param, target_param in zip(self.q_net.parameters(), self.t_net.parameters()):
                            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
                    else:
                        if trained_step % int(self.config.tau) == 0:
                            self.t_net.load_state_dict(self.q_net.state_dict())

                    ############### 保存模型 ###############
                    if trained_step % self.config.save_interval == 0:
                        torch.save(self.q_net.state_dict(),
                                   os.path.join(self.config.save_ckpt_path, 'ckpt_{}.pth'.format(trained_step)))

                if done or epoch_step >= self.config.epoch_steps - 1:
                    if self.config.noisy_net:
                        print('env: {}-{}, '
                              'epoch: {}, trained_step: {},'
                              ' loss: {}, steps: {}, reward: {}'.format(info['world'], info['stage'],
                                                                        epoch, trained_step,
                                                                        loss, epoch_step, epoch_reward))
                    else:
                        print('env: {}-{}, '
                              'epoch: {}, trained_step: {},'
                              ' loss: {}, steps: {}, greedy: {}, reward: {}'.format(info['world'], info['stage'],
                                                                                    epoch, trained_step,
                                                                                    loss, epoch_step, self.e_greedy,
                                                                                    epoch_reward))
                    break

                state = next_state
            self.e_greedy = max(self.config.e_greedy_end, self.e_greedy + self.e_greedy_decay_per_epoch)

    def _sample_batch_data(self):
        ############ 采样batch_size组n-step数据 ############
        bias_weight = None
        if self.config.prioritized_replay:
            batch_idx, batch_seq_data, sampling_weight = self.replay_buffer.sample(self.config.batch_size, self.config.n_step_learning)
            ############ 计算重要性采样补偿系数 ############
            bias_max_weight = (len(self.replay_buffer) * self.replay_buffer.min_priority) ** (-self.prioritized_beta)
            bias_weight = (len(self.replay_buffer) * np.array(sampling_weight)) ** (-self.prioritized_beta)
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
        batch_done = stack_data([data[4] for data in batch_data], self.device).to(batch_reward.dtype)
        batch_n_step = [data[5] for data in batch_data]
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_n_step, batch_idx, bias_weight

    def _one_batch_train(self):
        ############ 构造batch数据 ############
        state, action, reward, next_state, done, n_step, batch_idx, bias_weight = self._sample_batch_data()

        ############### 计算delta_z ###############
        if self.config.distributional_atom_size > 1:
            delta_z = float(self.config.distributional_v_max - self.config.distributional_v_min) / (self.config.distributional_atom_size - 1)

        with torch.no_grad():
            ############### 按n-step公式计算gamma ###############
            gamma = stack_data([self.config.gamma ** n for n in n_step], self.device)

            ############### 选择next_action ###############
            if self.config.double_q:
                t_action = torch.argmax(self.q_net(next_state), dim=-1)
            else:
                t_action = torch.argmax(self.t_net(next_state), dim=-1)

            ############### 计算目标价值 ###############
            if self.config.distributional_atom_size > 1:
                ############### 计算next_action的价值分布概率 ###############
                t_dist = self.t_net(next_state, return_dist=True)
                t_dist = t_dist[range(self.config.batch_size), t_action]
                ############### 阶段1：价值迭代 ###############
                t_z = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * gamma.unsqueeze(1) * self.t_net.support_z
                t_z = t_z.clamp(self.config.distributional_v_min, self.config.distributional_v_max)
                ############### 阶段2：价值投影 ###############
                b = (t_z - self.config.distributional_v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()
                offset = torch.linspace(0, (self.config.batch_size - 1) * self.config.distributional_atom_size, self.config.batch_size) \
                    .long().unsqueeze(1).expand(self.config.batch_size, self.config.distributional_atom_size).to(self.device)
                proj_dist = torch.zeros(t_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(0,
                                              (l + offset).view(-1),
                                              (t_dist * (u.float() - b)).view(-1))
                proj_dist.view(-1).index_add_(0,
                                              (u + offset).view(-1),
                                              (t_dist * (b - l.float())).view(-1))
            else:
                q_observation = reward + (1 - done) * gamma * self.t_net(next_state).gather(1, t_action.unsqueeze(1)).squeeze(1)

        ############### 计算loss ###############
        if self.config.distributional_atom_size > 1:
            ############### CrossEntropyLoss ###############
            dist = self.q_net(state, return_dist=True)
            log_p = torch.log(dist[range(self.config.batch_size), action])
            loss = -(proj_dist * log_p).sum(1)
        else:
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
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        return loss.cpu().item()

    def play(self, ckpt_step, reset_noop=-1, clear_noisy=False, reset_noisy=False, verbose=True, export_video_fn=None):
        ############### VideoWriter既能显示、又能保存video ###############
        from CleanRL.common.video_writer import VideoWriter
        video_writer = VideoWriter()
        assert self.env.render_mode == 'rgb_array', 'when using VideoWriter to preview images, please set render_mode to "rgb_array"'
        ############### 加载训练后的模型 ###############
        ckpt_fn = os.path.join(self.config.save_ckpt_path, 'ckpt_{}.pth'.format(ckpt_step))
        self.q_net.load_state_dict(torch.load(ckpt_fn, map_location='cpu'))
        self.q_net.to('cpu')
        self.q_net.eval()
        if self.config.noisy_net:
            assert not clear_noisy or not reset_noisy, 'clear_noisy和reset_noisy不能同时为True'
            if clear_noisy:
                self.q_net.zero_noise()
            if reset_noisy:
                self.q_net.reset_noise()

        state, _ = self.env.reset(reset_noop)
        total_reward = 0
        step = 0
        while True:
            video_writer.add_frame(self.env.render())
            step += 1
            ############### 选择action ###############
            q_values = self.q_net(stack_data([state, ]))
            action = torch.argmax(q_values).cpu().item()  # 模型最优动作

            ############### 执行action ###############
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            is_done = terminated or truncated

            if verbose:
                print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
                time.sleep(1 / 24)

            if is_done:
                print('step: {}, reward:{}, total_reward: {}'.format(step, reward, total_reward))
                if verbose:
                    print('game over')
                    time.sleep(2)
                break

        video_writer.flush(export_video_fn)


if __name__ == "__main__":
    pass
