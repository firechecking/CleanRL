# -*- coding: utf-8 -*-
# @Time    : 2024/5/1 10:54
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : trpo.py
# @Software: CleanRL
# @Description: trpo

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


class TRPOConfig():
    def __init__(self, **kwargs):
        self.epoches = 2000
        self.epoch_steps = 200
        self.gamma = 0.9
        self.lr_critic = 0.01
        self.load_path = './ckpt.pt'
        self.n_steps = self.epoch_steps  # 实验下来完整采样一轮数据后进行训练的效果更好

        self.gae_lambda = 0.9
        self.kl_constraint = 0.00005
        self.line_search_alpha = 0.5

        for k, v in kwargs.items():
            setattr(self, k, v)


class TRPO():
    def __init__(self, env, actor, critic, config: TRPOConfig):
        self.env = env
        self.actor, self.critic = actor, critic
        self.config = config

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
                    if epoch == self.config.epoch_steps:  # 使用unwrapped env，需要手动标记终止状态（终止状态的未来价值为0）
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

    def compute_surrogate_obj(self, states, actions, advantages, old_log_probs, actor):
        # E_s[E_a[(pi_new/pi_old)*A]]
        # 1. 重要性采样系数乘以优势度
        # 2. 用均值来代替期望
        mu, std = actor(states)
        action_dists = actor.distribution(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantages)

    def fisher_matrix_vector_product(self,
                                     states,
                                     old_action_dists,
                                     vector,
                                     damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        ############ KL散度的期望 ############
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists,
                                                             new_action_dists))
        ############ KL散度一阶导 ############
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        ############ 一阶导点乘v向量 ############
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        ############ 点乘后求二阶导 ############
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_parameters = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):
            ############ 从最大步长开始，逐步向下搜索 ############
            coef = self.config.line_search_alpha ** i
            new_parameters = old_parameters + coef * max_vec

            ############ 参数更新之后的新策略 ############
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_parameters, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)

            ############ 参数更新前后，策略的kl散度的期望(均值) ############
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            ############ 参数更新后的目标函数值 ############
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)

            ############ 如果目标函数有提升，并且满足kl散度约束，则接受更新，否在向下搜索更小步长 ############
            if new_obj > old_obj and kl_div < self.config.kl_constraint:
                return new_parameters

        ############ 如果最小步长都不满足条件，则放弃本次更新 ############
        return old_parameters

    def _one_batch_train(self, batch_data):
        with CodeBlock("构造batch数据"):
            states = stack_data([data[0] for data in batch_data]).to(torch.float32)
            actions = stack_data([data[1] for data in batch_data]).view(-1, 1).to(torch.float32)
            rewards = stack_data([data[2] for data in batch_data]).view(-1, 1).to(torch.float32)
            next_states = stack_data([data[3] for data in batch_data]).to(torch.float32)
            dones = stack_data([data[4] for data in batch_data]).view(-1, 1).to(torch.int)

        with CodeBlock("Critic更新"):
            rewards = (rewards + 8.0) / 8.0  # 对Pendulum-v1任务的奖励进行修改,方便训练
            td_targets = rewards + self.config.gamma * self.critic(next_states) * (1 - dones)

        with CodeBlock("计算目标函数"):
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
                ############ 近似目标函数：E_s[E_a[(pi_new/pi_old)*A]] ############
                # 封装成函数，方便之后的line_search复用
                surrogate_obj = self.compute_surrogate_obj(states, actions, advantages, old_log_probs, self.actor)

            with CodeBlock("优化步长、优化方向"):
                ############ 计算一阶导g ############
                grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
                grads = torch.cat([grad.view(-1) for grad in grads]).detach()

                ############ 共轭梯度求解F^(-1)g ############
                x = torch.zeros_like(grads)
                r = -grads.clone()
                d = -r.clone()
                r_dot_r = torch.dot(r, r)
                for i in range(10):
                    F_dot_d = self.fisher_matrix_vector_product(states, old_action_dists, d)
                    alpha = r_dot_r / torch.dot(d, F_dot_d)
                    x += alpha * d
                    r += alpha * F_dot_d
                    r_dot_r_new = torch.dot(r, r)
                    if r_dot_r_new < 1e-10:
                        break
                    d = -r + (r_dot_r_new / r_dot_r) * d
                    r_dot_r = r_dot_r_new

                ############ 更新方向 ############
                descent_direction = x
                ############ 最大更新步长 ############
                F_dot_x = self.fisher_matrix_vector_product(states, old_action_dists, x)
                max_step = torch.sqrt(2 * self.config.kl_constraint / (torch.dot(x, F_dot_x) + 1e-8))

            with CodeBlock("actor更新步长：线性搜索"):
                new_actor_parameters = self.line_search(states, actions, advantages, old_log_probs, old_action_dists, descent_direction * max_step)

        with CodeBlock("应用一轮参数更新"):
            ############ critic更新 ############
            critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic(states), td_targets.detach()))
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            ############ actor更新 ############
            torch.nn.utils.convert_parameters.vector_to_parameters(new_actor_parameters, self.actor.parameters())

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
