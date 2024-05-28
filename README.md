# CleanRL

项目特点

* **依赖最小化**：除了python基础库和pytorch基础运算外，不借助其他第三方库
* **理论+代码**：包含几乎所有强化学习相关的原理和推导，并从零实现常见的强化学习算法
* **实践案例**：包含AlphaZero、RLHF等热门、最新的实践案例
* **代码简洁**：参考原论文及多种开源实现，尽量简化代码，确保正确的同时降低学习难度

**欢迎大家来一起完善代码和教程**

文字教程见：

* [知乎: 从零实现强化学习、RLHF、AlphaZero](https://zhuanlan.zhihu.com/p/672591581)

## 已更新文字教程

* [基于价值的强化学习1](https://zhuanlan.zhihu.com/p/673543350)
    * 强化学习中的**蒙特卡洛法**、**时序差分法**、**贝尔曼方程**的对比及代码实现
    * **q-learning**原理、代码实现
    * **sarsa**、**sarsa-lambda**原理、代码实现
    * **DQN**原理、代码实现
    * DQN优化1：**Replay Buffer**、**Fixed Q Target**
    * DQN优化2：**Double DQN**
    * DQN优化3：**Dueling DQN**
    * DQN优化4：**Prioritized Replay Buffer**
    * **Sum-Tree**原理、代码实现
* [基于价值的强化学习2](https://zhuanlan.zhihu.com/p/677135856)
    * DQN中六种优化方法的直观对比
    * DQN优化5：**Multi-step Learning**
    * DQN优化6：**Noisy Net**
    * DQN优化7：**Distributional RL**
    * Rainbow实践案例：super-mario训练（含环境、键盘demo、算法、checkpoint等）
* 基于策略的强化学习1

## 已更新算法代码

* [Monte-Carlo](Experiments/run_mc_cartpole.py)
* [Q-Learning](CleanRL/algorithms/q_learning)
* [Sarsa（及Sarsa-Lambda）](CleanRL/algorithms/sarsa)
* [DQN（及Rainbow）](CleanRL/algorithms/dqn)
* [SuperMario with Rainbow*](Examples/super_mario)
* [REINFORCE](CleanRL/algorithms/reinforce)
* [AC, A2C](CleanRL/algorithms/ac_a2c)
* [A3C](CleanRL/algorithms/a3c)
* [DPG](CleanRL/algorithms/dpg_ddpg)
* [DDPG](CleanRL/algorithms/dpg_ddpg)
* [TRPO](CleanRL/algorithms/trpo)
* PPO

## TODO

- [x] 基于价值的强化学习-算法实现
- [x] 基于价值的强化学习-实践案例 (super-mario)
- [x] 基于策略的强化学习-算法实现
- [ ] 基于策略的强化学习-实践案例 (机械臂/机械狗)
- [ ] RLHF: 通义千问模型的ppo、dpo训练
- [ ] Alpha: AlphaZero、MuZero训练五子棋、斗地主、麻将