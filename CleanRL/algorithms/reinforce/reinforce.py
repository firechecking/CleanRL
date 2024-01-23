# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 20:09
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : reinforce.py.py
# @Software: CleanRL
# @Description: reinforce.py

import os, time, math
import numpy as np
import torch


def stack_data(batch_data):
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data)
    else:
        return torch.tensor(np.array(batch_data))


class REINFORCEConfig():
    def __init__(self, **kwargs):
        self.epoches = 5000
        self.epoch_steps = 500
        self.gamma = 0.9
        self.lr = 1e-3
        self.load_path = 'ckpt.pt'
        self.save_interval = 10

        self.continuous_action = False

        for k, v in kwargs.items():
            setattr(self, k, v)


class REINFORCE():
    def __init__(self, env, policy_net, config: REINFORCEConfig):
        self.env = env
        self.policy_net = policy_net
        self.config = config

        if self.config.load_path and os.path.exists(self.config.load_path):
            self.policy_net.load_state_dict(torch.load(self.config.load_path))

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

    def learn(self):
        print('start learn...')
        for epoch in range(1, self.config.epoches + 1):
            ############### 重置环境 ###############
            state, _ = self.env.reset()
            epoch_reward = 0
            replay_buffer = []  # 每局游戏都清空replay_buffer
            for epoch_step in range(1, self.config.epoch_steps + 1):
                ############### 选择action ###############
                with torch.no_grad():
                    if self.config.continuous_action:
                        mean, deviation = self.policy_net(stack_data([state, ]))
                        dist = self.policy_net.distribution(mean.view(1, ).data, deviation.view(1, ).data)
                    else:
                        probs = self.policy_net(stack_data([state, ]))[0]
                        dist = self.policy_net.distribution(probs)

                    action = dist.sample().numpy()

                    if self.config.continuous_action:
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

                ############### 执行action ###############
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # reward = (reward + 8.1) / 8.1  # 将reward投影到[-1, 1]
                done = terminated or truncated
                replay_buffer.append((state, action, reward))
                epoch_reward += reward

                state = next_state

                if done or epoch_step == self.config.epoch_steps:
                    ############ 一局结束再更新 ############
                    loss = self._one_batch_train(replay_buffer)
                    print('epoch: {}, steps: {}, total_reward: {}, loss: {}'.format(epoch, epoch_step, epoch_reward, loss))

                    if epoch % self.config.save_interval == 0:
                        torch.save(self.policy_net.state_dict(), self.config.load_path)
                    break

    def _one_batch_train(self, replay_buffer):
        state = stack_data([data[0] for data in replay_buffer])
        action = stack_data([data[1] for data in replay_buffer])
        reward = stack_data([data[2] for data in replay_buffer])

        ############ 计算回合内单步收益 ############
        discounted_reward = torch.zeros((len(replay_buffer), 1))
        running_add = 0
        for t in reversed(range(0, len(replay_buffer))):
            running_add = running_add * self.config.gamma + reward[t]
            discounted_reward[t][0] = running_add

        ############ 减去基线 ############
        discounted_reward -= torch.mean(discounted_reward)
        discounted_reward /= torch.std(discounted_reward)

        ############ CrossEntropy ############
        if self.config.continuous_action:
            mean, deviation = self.policy_net(state)
            dist = self.policy_net.distribution(mean, deviation)
        else:
            probs = self.policy_net(state)
            dist = self.policy_net.distribution(probs)
        ce = -dist.log_prob(action)

        ############ loss ############
        loss = ce.view(-1) * discounted_reward.view(-1)  # discounted_reward为正时，加速向期望方向优化；discounted_reward为负时，向反方向优化
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def play(self):
        print('start play...')
        ############### 加载训练后的模型 ###############
        self.policy_net.load_state_dict(torch.load(self.config.load_path, map_location='cpu'))
        self.policy_net.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            if self.config.continuous_action:
                mean, deviation = self.policy_net(stack_data([state, ]))
                dist = self.policy_net.distribution(mean.view(1, ).data, deviation.view(1, ).data)
            else:
                probs = self.policy_net(stack_data([state, ]))[0]
                dist = self.policy_net.distribution(probs)
            action = dist.sample().numpy()
            if self.config.continuous_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

            ############### 执行action ###############
            state, reward, terminated, truncated, _ = self.env.step(action)
            # reward = (reward + 8.1) / 8.1
            total_reward += reward
            is_done = terminated or truncated

            print('step: {}, action: {}, total reward: {}'.format(step, action, total_reward))
            if is_done:
                time.sleep(2)
                break


if __name__ == "__main__":
    pass
