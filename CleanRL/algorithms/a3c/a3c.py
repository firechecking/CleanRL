# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 19:56
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : a3c.py
# @Software: CleanRL
# @Description: a3c

import os, time
import multiprocessing as mp
import numpy as np
import torch


def stack_data(batch_data):
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data)
    else:
        return torch.tensor(np.array(batch_data))


class A3CConfig():
    def __init__(self, **kwargs):
        self.epoches = 5000
        self.epoch_steps = 500
        self.gamma = 0.9
        self.lr = 1e-4
        self.load_path = './ckpt.pt'
        self.save_interval = 1

        self.workers = 8

        for k, v in kwargs.items():
            setattr(self, k, v)


def worker(worker_name, global_net, optimizer, env_and_net_builder, config: A3CConfig):
    print(f'start process {worker_name}: {os.getpid()}')

    env, local_net = env_and_net_builder()
    for epoch in range(1, config.epoches + 1):
        ############ 同步global_net参数至local_net ############
        local_net.load_state_dict(global_net.state_dict())

        ############### 重置环境 ###############
        state, _ = env.reset(seed=100000 * int(worker_name.split('_')[-1]) + epoch)
        average_reward, epoch_reward = None, 0

        for epoch_step in range(1, config.epoch_steps + 1):
            ############ 选择action ############
            with torch.no_grad():
                mean, deviation, value = local_net(stack_data([state, ]))
                dist = local_net.distribution(mean.view(1, ).data, deviation.view(1, ).data)
                action = dist.sample().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high).astype('float32')

            ############### 执行action ###############
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            epoch_reward += reward

            ############### 模型更新,并重新同步参数 ###############
            _one_step_train(global_net, local_net, optimizer, config, state, action, reward, next_state, done)
            local_net.load_state_dict(global_net.state_dict())

            state = next_state

            if done or epoch_step == config.epoch_steps:
                if worker_name == 'worker_0':
                    print('epoch: {}, steps: {}, total_reward: {}'.format(epoch, epoch_step, epoch_reward))
                    if epoch % config.save_interval == 0:
                        torch.save(global_net.state_dict(), config.load_path)
                break


def _one_step_train(global_net, local_net, optimizer, config, state, action, reward, next_state, done):
    ############ 重新forward一遍 ############
    mean, deviation, value = local_net(stack_data([state, ]))

    ############ 计算td_error和critic_loss ############
    if done:
        v_ = torch.tensor(0.).view(1, 1)
    else:
        with torch.no_grad():
            _, _, v_ = local_net(stack_data([next_state, ]))
    target_v = reward + config.gamma * v_
    td_error = target_v - value
    critic_loss = torch.square(td_error)

    ############ 计算actor_loss ############
    dist = local_net.distribution(mean, deviation)
    ce = -dist.log_prob(stack_data([action, ]))
    actor_loss = ce * td_error.detach()

    ############ 计算总的loss ############
    loss = critic_loss + actor_loss

    ############ optimizer是对global_net，所以需要手动清空local_net的grad ############
    for local_param in local_net.parameters():
        local_param.grad = None
    loss.backward()

    ############ 将grad从local_net同步到global_net ############
    optimizer.zero_grad()
    for global_param, local_param in zip(global_net.parameters(), local_net.parameters()):
        global_param.grad = local_param.grad
    optimizer.step()


class A3C():
    def __init__(self, env_and_net_builder, config: A3CConfig):
        self.env_and_net_builder = env_and_net_builder
        self.config = config

        ############### 使用pytorch的share_memory()启用共享内存 ###############
        _, self.global_net = env_and_net_builder()
        if os.path.exists(self.config.load_path):
            self.global_net.load_state_dict(torch.load(self.config.load_path))

        self.global_net.share_memory()

        self.optimizer = torch.optim.Adam(self.global_net.parameters(), lr=self.config.lr)

        ############### 启动多个工作进程 ###############
        self.processes = []
        for i in range(self.config.workers):
            p = mp.Process(target=worker,
                           args=(f'worker_{i}', self.global_net, self.optimizer, env_and_net_builder, self.config))
            self.processes.append(p)

    def learn(self):
        print('start learn...')
        for p in self.processes:
            p.start()
        for p in self.processes:
            p.join()

    def play(self):
        print('start play...')
        ############### 加载训练后的模型 ###############
        self.env, self.global_net = self.env_and_net_builder(render_model='human')
        self.global_net.load_state_dict(torch.load(self.config.load_path, map_location='cpu'))
        self.global_net.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            mean, deviation, _ = self.global_net(stack_data([state, ]))
            dist = self.global_net.distribution(mean.view(1, ).data, deviation.view(1, ).data)
            action = dist.sample().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

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
