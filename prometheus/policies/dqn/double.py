"""
=============================================
Double DQN 策略
=============================================

Double DQN 的核心思想：
解耦动作选择和价值评估，解决 DQN 过高估计 Q 值的问题。

标准 DQN:
- 用目标网络选择动作并评估价值
- target_q = target_network(next_state).max()

Double DQN:
- 用主网络选择动作，用目标网络评估价值
- next_action = policy_network(next_state).argmax()
- target_q = target_network(next_state)[next_action]

论文: Deep Reinforcement Learning with Double Q-learning (2016)
"""

import torch
from typing import Optional

from prometheus.policies.dqn.base import DQNPolicyBase, QNetwork


class DoubleDQNPolicy(DQNPolicyBase):
    """
    Double DQN 策略

    继承自 DQNPolicyBase，使用相同的 Q 网络结构。
    Double DQN 的差异在智能体的 learn 方法中实现。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        epsilon: float = 0.1,
        device: str = "auto"
    ):
        super().__init__(state_dim, action_dim, hidden_dim, epsilon, device)

    # Double DQN 的核心逻辑在 Agent 中实现，这里复用基类
