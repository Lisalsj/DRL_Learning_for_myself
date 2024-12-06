# RL
## 我眼中的RL
Agent 通过observe environment，根据现在的state 进行decision，然后选择了一个action 从而获得到 reward的过程
![RL示意图](../image/img.png "RL示意图")

## RL的小知识点
1. RL的输入样本是**序列数据**，这么看来，RL是面向过程的学习。
2. Agent获得能力的过程是一个探索和利用 **（exploration and exploitation）** 的过程。
3. RL 没有supervisor 一般是一个延迟的reward 来进行的。

# DRL
DRL = RL + DL
## 标准强化学习
需要先设计特征（需要人为定义或者选择特征），再通过设计分类网络或者价值估计函数来采取动作
## 深度强化学习
不需要设计特征，输入state 就可以得到action。可以通过一个神经网络来你和价值函数或策略网络

# 序列决策
历史是一个包含观测、动作、奖励的序列
&& H_t = o_1, a_1, r_1, \ldots, o_t, a_t, r_t &&