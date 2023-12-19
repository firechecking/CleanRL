# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 22:07
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : sum_tree.py
# @Software: CleanRL
# @Description: sum_tree

import random
from collections import defaultdict


class SumTree():
    def __init__(self, capacity):
        ############ 根据容量计算叶子节点数量(完全二叉树) ############
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.tree = defaultdict(float)
        self.dataset = {}

        self.leaf_idx_offset = self.capacity - 1  # 叶子节点开始位置
        self.tree_leaf_pos = 0  # 待插入的叶子结点位置

        self.length = 0
        self.max_priority = 0

    def add(self, data, priority=None):
        self.length = min(self.length + 1, self.capacity)

        ############ 新加入数据使用最高优先级 ############
        if priority is None:
            priority = self.max_priority
        elif priority > self.max_priority:
            self.max_priority = priority

        ############ 实现插入 ############
        leaf_real_idx = self.leaf_idx_offset + self.tree_leaf_pos
        self.dataset[leaf_real_idx] = data

        ############ 更新节点优先级 ############
        self.update_priority(leaf_real_idx, priority)

        ############ 更新下个叶子节点位置 ############
        self.tree_leaf_pos += 1
        self.tree_leaf_pos %= self.capacity

    def update_priority(self, leaf_real_idx, priority):
        ############ 更新叶子节点优先级 ############
        self.tree[leaf_real_idx] = priority

        ############ 循环更新父节点 ############
        while True:
            parent_idx = (leaf_real_idx - 1) // 2
            if leaf_real_idx % 2 == 0:
                self.tree[parent_idx] = self.tree[leaf_real_idx - 1] + self.tree[leaf_real_idx]
            else:
                self.tree[parent_idx] = self.tree[leaf_real_idx] + self.tree[leaf_real_idx + 1]
            leaf_real_idx = parent_idx
            if parent_idx == 0:
                break

    def sample(self, batch_size):
        ############ 分段 ############
        seg = self.tree[0] / batch_size
        batch_data_idx = []
        for i in range(batch_size):
            rnd_value = random.uniform(a=seg * i,
                                       b=seg * (i + 1))
            ############ 循环遍历子节点 ############
            son_idx = 1
            while True:
                if self.tree[son_idx] < rnd_value:
                    rnd_value -= self.tree[son_idx]
                    son_idx += 1

                if son_idx >= self.leaf_idx_offset:
                    ############ 已定位到叶子节点 ############
                    break

                ############ 更新下一个子节点位置 ############
                son_idx = son_idx * 2 + 1

            batch_data_idx.append(son_idx)

        return batch_data_idx, [self.dataset[idx] for idx in batch_data_idx]


if __name__ == "__main__":
    # 建树
    sum_tree = SumTree(3)
    for i in (3, 10, 12, 4, 32, 12, 13):
        sum_tree.add(i, i)


    def _sample():
        for i in range(sum_tree.capacity + sum_tree.leaf_idx_offset):
            if i >= sum_tree.leaf_idx_offset:
                print(f'{i}: prior-{sum_tree.tree[i]}, data-{sum_tree.dataset[i]}')
            else:
                print(f'{i}: prior-{sum_tree.tree[i]}')

        # 采样
        counter = defaultdict(int)
        total = 0
        for i in range(10000):
            for idx, v in zip(*sum_tree.sample(3)):
                counter[v] += 1
                total += 1

        # 查看采样概率
        for k, v in counter.items():
            print(f'{k}: {v / total}')


    print('=' * 30, '第1次采样', '=' * 30)
    _sample()

    print('=' * 30, '第2次采样', '=' * 30)
    for i in range(sum_tree.leaf_idx_offset, sum_tree.leaf_idx_offset + sum_tree.capacity):
        sum_tree.update_priority(i, 1)
    _sample()
