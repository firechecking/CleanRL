# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 19:06
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : arguments.py
# @Software: CleanRL
# @Description: arguments

import argparse


def init_args():
    parser = argparse.ArgumentParser(description='super mario dqn')

    group = parser.add_argument_group(title='common argument')
    group.add_argument('--epoches', default=1000, type=int)
    group.add_argument('--epoch_steps', default=500, type=int)
    group.add_argument('--batch_size', default=512, type=int)
    group.add_argument('--e_greedy_start', default=0.95, type=float)
    group.add_argument('--e_greedy_end', default=0.1, type=float)
    group.add_argument('--e_greedy_decay', default=500, type=int)
    group.add_argument('--gamma', default=0.9, type=float)
    group.add_argument('--lr', default=1e-4, type=float)
    group.add_argument('--save_interval', default=100, type=int)
    group.add_argument('--load_path', default=None, type=str)
    group.add_argument('--save_ckpt_path', default='./q_net/', type=str)
    group.add_argument('--grad_clip_norm', default=1.0, type=float)

    group = parser.add_argument_group(title='parameters for replay_buffer and target_net')
    group.add_argument('--replay_buffer_size', default=100000, type=int)
    group.add_argument('--replay_warm_up', default=1000, type=int)
    group.add_argument('--tau', default=0.001, type=float)

    group = parser.add_argument_group(title='parameters for dueling, double, prioritized')
    group.add_argument('--no_dueling', dest='dueling', default=True, action='store_false')
    group.add_argument('--no_double_q', dest='double_q', default=True, action='store_false')
    group.add_argument('--no_prioritized_replay', dest='prioritized_replay', default=True, action='store_false')
    group.add_argument('--prioritized_alpha', default=0.6, type=float)
    group.add_argument('--prioritized_beta', default=0.4, type=float)

    group = parser.add_argument_group(title='parameters for n_step, noisy')
    group.add_argument('--n_step_learning', default=3, type=int)
    group.add_argument('--no_noisy_net', dest='noisy_net', default=True, action='store_false')

    group = parser.add_argument_group(title='parameters for distributional learning')
    group.add_argument('--distributional_atom_size', default=51, type=int)
    group.add_argument('--distributional_v_min', default=-10, type=int)
    group.add_argument('--distributional_v_max', default=100, type=int)

    return parser


if __name__ == "__main__":
    pass
