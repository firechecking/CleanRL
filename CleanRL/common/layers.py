# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 20:36
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : layers.py
# @Software: CleanRL
# @Description: layers

import math
import torch


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init

        self.w_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.b_sigma = torch.nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('w_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('b_epsilon', torch.Tensor(out_features))

        self.init_parameters()
        self.reset_noise()

    def forward(self, x):
        return torch.nn.functional.linear(x,
                                          self.w_mu + self.w_sigma * self.w_epsilon,
                                          self.b_mu + self.b_sigma * self.b_epsilon)

    def init_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.b_mu.data.uniform_(-mu_range, mu_range)

        self.w_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.b_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _noise_func(self, size):
        ############### f(x) = sgn(x)·sqrt(|x|) ###############
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        ############### Factorised Gaussian noise: https://arxiv.org/abs/1706.10295 ###############
        in_epsilon = self._noise_func(self.in_features)
        out_epsilon = self._noise_func(self.out_features)

        self.w_epsilon.copy_(out_epsilon.ger(in_epsilon))
        self.b_epsilon.copy_(self.out_features)


if __name__ == "__main__":
    pass
