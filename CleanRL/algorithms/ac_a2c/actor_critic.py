# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 23:09
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : actor_critic.py
# @Software: CleanRL
# @Description: actor_critic

import os, time
import numpy as np
import torch


def stack_data(batch_data):
    assert isinstance(batch_data, (list, tuple))
    if isinstance(batch_data[0], torch.Tensor):
        return torch.stack(batch_data)
    else:
        return torch.tensor(np.array(batch_data))


class ActorCriticConfig():
    def __init__(self, **kwargs):
        self.epoches = 1000
        self.epoch_steps = 500
        self.gamma = 0.9
        self.lr_actor = 0.001
        self.lr_critic = 0.01
        self.load_path = './ckpt.pt'
        self.save_interval = 1

        self.a2c = True

        for k, v in kwargs.items():
            setattr(self, k, v)


class ActorCritic():
    def __init__(self, env, actor, critic, config: ActorCriticConfig):
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

    def learn(self):
        print('start learn...')
        for epoch in range(1, self.config.epoches + 1):
            state, _ = self.env.reset()
            average_reward, epoch_reward = None, 0
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

                ############### 模型更新 ###############
                self._one_step_train(state, action, reward, next_state)

                state = next_state

                if done or epoch_step == self.config.epoch_steps:
                    print('epoch: {}, steps: {}, total_reward: {}'.format(epoch, epoch_step, epoch_reward))
                    if epoch % self.config.save_interval == 0:
                        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()},
                                   self.config.load_path)
                    break

    def _one_step_train(self, state, action, reward, next_state):
        ############ 训练critic ############
        with torch.no_grad():
            v_ = self.critic(stack_data([next_state, ]))
        target_v = reward + self.config.gamma * v_
        td_error = target_v - self.critic(stack_data([state, ]))
        loss = torch.square(td_error).mean()

        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        ############ 训练actor ############
        mean, deviation = self.actor(stack_data([state, ]))
        dist = self.actor.distribution(mean, deviation)
        ce = -dist.log_prob(stack_data([action, ]))

        ############ ac, a2c ############
        if self.config.a2c:
            loss = ce * td_error.detach()
        else:
            loss = ce * target_v.detach()

        self.optimizer_actor.zero_grad()
        loss.mean().backward()
        self.optimizer_actor.step()

    def play(self):
        print('start play...')
        ############### 加载训练后的模型 ###############
        self.actor.load_state_dict(torch.load(self.config.load_path, map_location='cpu')['actor'])
        self.actor.eval()

        state, _ = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            ############### 选择action ###############
            mean, deviation = self.actor(stack_data([state, ]))
            dist = self.actor.distribution(mean.view(1, ).data, deviation.view(1, ).data)
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
