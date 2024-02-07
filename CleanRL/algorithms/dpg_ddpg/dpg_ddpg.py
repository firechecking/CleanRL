# -*- coding: utf-8 -*-
# @Time    : 2024/2/7 12:01
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : dpg_ddpg.py
# @Software: CleanRL
# @Description: dpg_ddpg

import os, time
import numpy as np
import torch


def stack_data(batch_data):
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data)
    else:
        return torch.tensor(np.array(batch_data))


class DPGConfig():
    def __init__(self, **kwargs):
        self.epoches = 1000
        self.epoch_steps = 500
        self.gamma = 0.9
        self.lr_actor = 0.001
        self.lr_critic = 0.01
        self.load_path = './ckpt.pt'
        self.save_interval = 1

        self.action_noise_start = 2
        self.action_noise_end = 0.
        self.action_noise_decay = 200

        for k, v in kwargs.items():
            setattr(self, k, v)


class DPG():
    def __init__(self, env, actor, critic, config: DPGConfig):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.config = config

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

        if os.path.exists(self.config.load_path):
            ckpt = torch.load(self.config.load_path)
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])

        self.action_noise = self.config.action_noise_start
        self.action_noise_decay_per_step = (self.config.action_noise_end - self.config.action_noise_start) / self.config.action_noise_decay

    def learn(self):
        print('start learn...')
        for epoch in range(1, self.config.epoches + 1):
            state, _ = self.env.reset()
            average_reward, epoch_reward = None, 0
            for epoch_step in range(1, self.config.epoch_steps + 1):
                ############ 选择action ############
                with torch.no_grad():
                    action = self.actor(stack_data([state, ])).cpu().view(-1)
                    action = np.random.normal(action.numpy(), self.action_noise, 1)
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype('float32')

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                epoch_reward += reward

                self._one_step_train(state, action, reward, next_state)

                state = next_state

                if done or epoch_step == self.config.epoch_steps:
                    print('epoch: {}, action_noise: {}, steps: {}, total_reward: {}'.format(epoch, self.action_noise, epoch_step, epoch_reward))
                    if epoch % self.config.save_interval == 0:
                        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()},
                                   self.config.load_path)

                    self.action_noise = max(self.action_noise + self.action_noise_decay_per_step, self.config.action_noise_end)
                    break

    def _one_step_train(self, state, action, reward, next_state):
        state = stack_data([state, ])
        action = stack_data([action, ])
        next_state = stack_data([next_state, ])
        ############ 训练critic ############
        with torch.no_grad():
            next_action = self.actor(next_state)
            v_ = self.critic(next_state, next_action)
        target_v = reward + self.config.gamma * v_
        td_error = target_v - self.critic(state, action)
        loss = torch.square(td_error).mean()

        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        ############ 训练actor ############
        self.critic.eval()
        action = self.actor(state)
        loss = -self.critic(state, action)

        self.optimizer_actor.zero_grad()
        loss.mean().backward()
        self.optimizer_actor.step()
        self.critic.train()

    def play(self):
        print('start play...')
        self.actor.load_state_dict(torch.load(self.config.load_path, map_location='cpu')['actor'])
        self.actor.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            action = self.actor(stack_data([state, ])).cpu().view(-1)
            action = action.detach().numpy()

            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            print('step: {}, action: {}, reward: {}, total reward: {}'.format(step, action, reward, total_reward))
            if done:
                time.sleep(2)
                break


if __name__ == "__main__":
    pass
