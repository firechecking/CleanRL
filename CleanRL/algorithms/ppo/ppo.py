# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 18:36
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : ppo.py
# @Software: CleanRL
# @Description: ppo.py

import os, copy, time
from contextlib import contextmanager
import numpy as np
import torch


def stack_data(batch_data):
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data)
    else:
        return torch.tensor(np.array(batch_data))


@contextmanager
def CodeBlock(name):
    yield


class PPOConfig():
    def __init__(self, **kwargs):
        self.epoches = 2000
        self.epoch_steps = 200
        self.gamma = 0.9
        self.lr_critic = 0.001
        self.lr_actor = 0.0001
        self.load_path = './ckpt.pt'
        self.n_steps = self.epoch_steps  # 实验下来完整采样一轮数据后进行训练的效果更好

        self.gae_lambda = 0.9

        self.ppo_type = 'clip'  # 支持'clip'、'penalty'，论文中说clip方式效果更好
        self.update_steps = 10
        self.clip_epsilon = 0.2
        self.penalty_target = 0.01
        self.penalty_beta = 3

        for k, v in kwargs.items():
            setattr(self, k, v)


class PPO():
    def __init__(self, env, actor, critic, config: PPOConfig):
        self.env = env
        self.actor, self.critic = actor, critic
        self.config = config

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.config.lr_critic)

        if os.path.exists(self.config.load_path):
            ckpt = torch.load(self.config.load_path)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])

    def learn(self):
        print('start learn...')
        for epoch in range(1, self.config.epoches + 1):
            state, _ = self.env.reset()
            epoch_reward = 0
            replay_buffer = []

            for epoch_step in range(1, self.config.epoch_steps + 1):
                ############ 选择action ############
                with torch.no_grad():
                    mean, deviation = self.actor(stack_data([state, ]))
                    dist = self.actor.distribution(mean.view(1, ).data, deviation.view(1, ).data)
                    action = dist.sample().numpy()
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

                ############### 执行action ###############
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                epoch_reward += reward
                replay_buffer.append([state, action, reward, next_state, done])

                ############### 模型训练 ###############
                if len(replay_buffer) == self.config.n_steps or done or epoch_step == self.config.epoch_steps:
                    if epoch_step == self.config.epoch_steps:  # 使用unwrapped env，需要手动标记终止状态（终止状态的未来价值为0）
                        replay_buffer[-1][-1] = True
                    ############### 对n_steps数据进行训练 ###############
                    self._one_batch_train(replay_buffer)
                    replay_buffer = []

                if done:
                    break

                state = next_state

            ############### 模型保存 ###############
            print('epoch:{}, epoch_step:{}, epoch_reward:{}'.format(epoch, epoch_step, epoch_reward))
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()
            }, self.config.load_path)

    def compute_surrogate_obj(self, states, actions, advantages, old_log_probs, actor, return_ratio=False):
        # 代码和TRPO相同，增加return_ratio参数直接返回pi_new/pi_old
        # E_s[E_a[(pi_new/pi_old)*A]]
        # 1. 重要性采样系数乘以优势度
        # 2. 用均值来代替期望
        mu, std = actor(states)
        action_dists = actor.distribution(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        if return_ratio:
            return ratio, action_dists
        return torch.mean(ratio * advantages)

    def _one_batch_train(self, batch_data):
        with CodeBlock("构造batch数据"):
            states = stack_data([data[0] for data in batch_data]).to(torch.float32)
            actions = stack_data([data[1] for data in batch_data]).view(-1, 1).to(torch.float32)
            rewards = stack_data([data[2] for data in batch_data]).view(-1, 1).to(torch.float32)
            next_states = stack_data([data[3] for data in batch_data]).to(torch.float32)
            dones = stack_data([data[4] for data in batch_data]).view(-1, 1).to(torch.int)

        with CodeBlock("计算目标函数"):
            rewards = (rewards + 8.0) / 8.0  # 对Pendulum-v1任务的奖励进行修改,方便训练
            td_targets = rewards + self.config.gamma * self.critic(next_states) * (1 - dones)
            with CodeBlock("优势度"):
                with torch.no_grad():
                    ############ 单步优势度 ############
                    # advantages = td_targets - self.critic(states)
                    ############ 广义优势度 ############
                    td_deltas = td_targets - self.critic(states)
                    td_deltas = td_deltas.detach().cpu().numpy()
                    advantages = []
                    _advantage = 0.0
                    for delta in td_deltas[::-1]:
                        _advantage = self.config.gamma * self.config.gae_lambda * _advantage + delta
                        advantages.append(_advantage)
                    advantages.reverse()
                    advantages = torch.tensor(advantages, dtype=torch.float32)

            with CodeBlock("近似目标函数"):
                with torch.no_grad():
                    ############ 采样策略的log概率：log π(a|s) ############
                    mu, std = self.actor(states)
                    old_action_dists = self.actor.distribution(mu, std)
                    old_log_probs = old_action_dists.log_prob(actions)

        with CodeBlock("PPO-update"):
            for _ in range(self.config.update_steps):
                ############ 计算：pi_new/pi_old ############
                ratio, new_action_dists = self.compute_surrogate_obj(states, actions, advantages, old_log_probs, self.actor, return_ratio=True)

                assert self.config.ppo_type in ['clip', 'penalty']
                if self.config.ppo_type == 'clip':
                    ############ 应用公式：min(r*A, clip(r, 1-epsilon, 1+epsilon)*A) ############
                    cliped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                    actor_loss = -torch.mean(torch.min(ratio * advantages, cliped_ratio * advantages))
                elif self.config.ppo_type == 'penalty':
                    ############ 带惩罚项的优化目标：E(ratio*A-beta*KL) ############
                    kl = torch.distributions.kl_divergence(old_action_dists, new_action_dists)
                    actor_loss = ratio * advantages - self.config.penalty_beta * kl
                    actor_loss = -torch.mean(actor_loss)
                    if kl.mean() > 4 * self.config.penalty_target:  # 如果kl散度过大，退出本轮更新循环
                        break

                ############ 更新actor ############
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                ############ 更新critic ############
                critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic(states), td_targets.detach()))
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            ############ (为训练下一次采样数据)更新kl惩罚项系数 ############
            if self.config.ppo_type == 'penalty':
                if kl.mean() < self.config.penalty_target / 1.5:
                    self.config.penalty_beta /= 2
                elif kl.mean() > self.config.penalty_target * 1.5:
                    self.config.penalty_beta *= 2
                self.config.penalty_beta = np.clip(self.config.penalty_beta, 1e-4, 10)  # 限制惩罚项系数范围

    def play(self):
        print('start eval...')
        if os.path.exists(self.config.load_path):
            ckpt = torch.load(self.config.load_path)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
        self.actor.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1

            mean, deviation = self.actor(stack_data([state, ]))
            m = self.actor.distribution(mean.view(1, ).data, 0.00001)
            # m = actor.distribution(mean.view(1, ).data, deviation.view(1, ).data)
            action = m.sample().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
            done = terminated or truncated
            if done:
                time.sleep(2)
                break


if __name__ == "__main__":
    pass
