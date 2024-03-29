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
        self.lrs = [1e-4, 5e-4]
        self.load_path = './ckpt.pt'
        self.save_interval = 1

        self.workers = 8
        self.shared_optimizer = True
        self.n_steps = 3
        self.objective_entropy = 0.005
        self.unique_net = True
        self.lock = True
        self.normalize_reward = True

        for k, v in kwargs.items():
            setattr(self, k, v)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, *args, **kwargs):
        super(SharedAdam, self).__init__(params, *args, **kwargs)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                ############ 启用共享内存 ############
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def worker(worker_name, lock, global_nets, optimizers, env_and_net_builder, config: A3CConfig):
    print(f'start process {worker_name}: {os.getpid()}')
    env, local_nets = env_and_net_builder(unique_net=config.unique_net)
    critic_loss, actor_loss = None, None
    for epoch in range(1, config.epoches + 1):
        ############ 同步global_net参数至local_net ############
        for local_net, global_net in zip(local_nets, global_nets):
            local_net.load_state_dict(global_net.state_dict())

        ############### 重置环境 ###############
        state, _ = env.reset(seed=100000 * int(worker_name.split('_')[-1]) + epoch)
        epoch_reward = 0
        replay_buffer = []

        for epoch_step in range(1, config.epoch_steps + 1):
            ############ 选择action ############
            with torch.no_grad():
                if config.unique_net:
                    mean, deviation = local_nets[0](stack_data([state, ]))
                else:
                    mean, deviation, value = local_nets[0](stack_data([state, ]))
                dist = local_nets[0].distribution(mean.view(1, ).data, deviation.view(1, ).data)
                action = dist.sample().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high).astype('float32')

            ############### 执行action ###############
            next_state, reward, terminated, truncated, _ = env.step(action)
            if config.normalize_reward:
                reward = (reward + 8.1) / 8.1
            done = terminated or truncated
            epoch_reward += reward
            replay_buffer.append((state, action, reward))

            ############### 模型更新,并重新同步参数 ###############
            if len(replay_buffer) == config.n_steps or done or epoch_step == config.epoch_steps:
                critic_loss, actor_loss = _one_step_train(lock, global_nets, local_nets, optimizers, config, replay_buffer, next_state, done)
                for local_net, global_net in zip(local_nets, global_nets):
                    local_net.load_state_dict(global_net.state_dict())
                replay_buffer = []

            state = next_state

            if done or epoch_step == config.epoch_steps:
                if worker_name == 'worker_0':
                    print('epoch: {}, steps: {}, critic_loss: {}, actor_loss: {}, epoch_reward: {}'.format(epoch, epoch_step, critic_loss, actor_loss,
                                                                                                           epoch_reward))
                    if epoch % config.save_interval == 0:
                        state_dict = {0: global_nets[0].state_dict()}
                        if config.unique_net:
                            state_dict[1] = global_nets[1].state_dict()
                        torch.save(state_dict, config.load_path)
                break


def _one_step_train(lock, global_nets, local_nets, optimizers, config, replay_buffer, next_state, done):
    ############ 重新forward一遍 ############
    batch_state = stack_data([data[0] for data in replay_buffer])
    if config.unique_net:
        mean, deviation = local_nets[0](batch_state)
        value = local_nets[1](batch_state)
    else:
        mean, deviation, value = local_nets[0](batch_state)

    ############ 估计最后一步的价值 ############
    if done:
        v_ = torch.tensor(0.).view(1, 1)
    else:
        with torch.no_grad():
            if config.unique_net:
                v_ = local_nets[1](stack_data([next_state, ]))
            else:
                _, _, v_ = local_nets[0](stack_data([next_state, ]))

    ############ 计算连续多步的价值 ############
    batch_v_target = []
    for _, _, _reward in reversed(replay_buffer):
        v_ = _reward + config.gamma * v_
        batch_v_target.append(v_)
    batch_v_target.reverse()

    ############ 计算td_error和critic_loss ############
    td_error = torch.cat(batch_v_target) - value
    critic_loss = torch.square(td_error).mean()

    ############ 计算actor_loss ############
    batch_action = stack_data([data[1] for data in replay_buffer])
    dist = local_nets[0].distribution(mean, deviation)
    ce = -dist.log_prob(batch_action)
    actor_loss = ce * td_error.detach() - config.objective_entropy * dist.entropy()
    actor_loss = actor_loss.mean()

    ############ optimizer是对global_net，所以需要手动清空local_net的grad ############
    for local_net in local_nets:
        for local_param in local_net.parameters():
            local_param.grad = None
    if config.unique_net:
        ############ 独立计算loss ############
        critic_loss.backward()
        actor_loss.backward()
    else:
        ############ 计算总的loss ############
        loss = critic_loss + actor_loss
        loss.backward()

    ############ 将grad从local_net同步到global_net ############
    if lock is not None and abs(critic_loss + actor_loss) > 100:
        lock.acquire()
    for optimizer in optimizers:
        optimizer.zero_grad()
    for local_net, global_net in zip(local_nets, global_nets):
        for global_param, local_param in zip(global_net.parameters(), local_net.parameters()):
            global_param.grad = local_param.grad
    for optimizer in optimizers:
        optimizer.step()
    if lock is not None and abs(critic_loss + actor_loss) > 100:
        lock.release()

    return critic_loss.cpu().item(), actor_loss.cpu().item()


class A3C():
    def __init__(self, env_and_net_builder, config: A3CConfig):
        self.env_and_net_builder = env_and_net_builder
        self.config = config

        ############### 使用pytorch的share_memory()启用共享内存 ###############
        _, self.global_nets = env_and_net_builder(unique_net=config.unique_net)
        if os.path.exists(self.config.load_path):
            ckpt = torch.load(self.config.load_path)
            for i, net in enumerate(self.global_nets):
                net.load_state_dict(ckpt[i])
                net.share_memory()

        if self.config.shared_optimizer:
            self.optimizers = [SharedAdam(net.parameters(), lr=self.config.lrs[i]) for i, net in enumerate(self.global_nets)]
        else:
            self.optimizers = [torch.optim.Adam(net.parameters(), lr=self.config.lrs[i]) for i, net in enumerate(self.global_nets)]

        ############### 启动多个工作进程 ###############
        self.lock = mp.Lock() if self.config.lock else None
        self.processes = []
        for i in range(self.config.workers):
            p = mp.Process(target=worker,
                           args=(f'worker_{i}', self.lock, self.global_nets, self.optimizers, env_and_net_builder, self.config))
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
        self.env, self.global_nets = self.env_and_net_builder(render_model='human', unique_net=self.config.unique_net)
        ckpt = torch.load(self.config.load_path)
        for i, net in enumerate(self.global_nets):
            net.load_state_dict(ckpt[i])
            net.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            if self.config.unique_net:
                mean, deviation = self.global_nets[0](stack_data([state, ]))
            else:
                mean, deviation, _ = self.global_nets[0](stack_data([state, ]))
            dist = self.global_nets[0].distribution(mean.view(1, ).data, 0.00001)  # play阶段可以将方差调小减少抖动
            action = dist.sample().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

            ############### 执行action ###############
            state, reward, terminated, truncated, _ = self.env.step(action)
            if self.config.normalize_reward:
                reward = (reward + 8.1) / 8.1
            total_reward += reward
            is_done = terminated or truncated

            print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
            if is_done:
                time.sleep(2)
                break


if __name__ == "__main__":
    pass
