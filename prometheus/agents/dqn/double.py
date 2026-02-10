"""
=============================================
Double DQN 智能体
=============================================

Double DQN 解决了标准 DQN 过高估计 Q 值的问题。

核心改动: compute_target_q 方法
- 标准 DQN: target_network(next_states).max()
- Double DQN: target_network(next_states)[policy_network(next_states).argmax()]
"""

import torch
import numpy as np
from typing import Dict

from prometheus.agents.dqn.base import DQNAgentBase, DQNAgent
from prometheus.policies.dqn.double import DoubleDQNPolicy
from prometheus.core import Config


class DoubleDQNAgent(DQNAgentBase):
    """
    Double DQN 智能体

    相比标准 DQN，只在 compute_target_q 方法上有差异：
    - 用主网络选择下一个动作
    - 用目标网络评估该动作的价值

    通俗解释：
    -----------
    标准 DQN 就像一个"贪心"的学生，总是用同一本书（目标网络）来
    选择答案并评分，容易产生"盲目自信"（过高估计）。

    Double DQN 像是用两本书，一本（主网络）用来选择答案，
    另一本（目标网络）用来评分，更客观公正。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config = None,
        device: str = "auto"
    ):
        """
        初始化 Double DQN 智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            config: 配置对象
            device: 计算设备
        """
        super().__init__(state_dim, action_dim, config, device)

        # 替换为 Double DQN 策略（实际上与基类相同，但类型更清晰）
        self.policy = DoubleDQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            epsilon=config.EPSILON_START if config else Config().EPSILON_START,
            device=device
        )
        # 重新创建目标网络（因为上面的 super().__init__ 创建的是旧的）
        from prometheus.policies.dqn.base import QNetwork
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim=128)
        self.target_network = self.target_network.to(self.policy.device)
        self.target_network.load_state_dict(self.policy.q_network.state_dict())

    def compute_target_q(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Double DQN 的核心：解耦动作选择和价值评估

        标准 DQN:
            next_q_values = target_network(next_states)
            target_q = reward + gamma * next_q_values.max()

        Double DQN:
            # 用主网络选择动作
            next_actions = policy_network(next_states).argmax()
            # 用目标网络评估
            next_q_values = target_network(next_states)
            target_q = reward + gamma * next_q_values[next_actions]

        Args:
            next_states: 下一状态批次
            rewards: 奖励批次
            dones: 结束标记批次

        Returns:
            目标 Q 值
        """
        # === Double DQN: 解耦选择和评估 ===
        with torch.no_grad():
            # 1. 用主网络选择下一个动作
            next_q_policy = self.policy.q_network(next_states)
            next_actions = next_q_policy.argmax(1)

            # 2. 用目标网络评估该动作的价值
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # 3. 计算 target Q
            target_q = rewards + self.config.GAMMA * next_q_values * (1 - dones)

        return target_q


# 保持向后兼容的别名
DoubleDQN = DoubleDQNAgent
