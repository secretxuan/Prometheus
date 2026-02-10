"""
=============================================
DQN 策略基类
=============================================
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from prometheus.policies.base import TorchPolicy


class QNetwork(nn.Module):
    """
    Q 网络 - 神经网络近似 Q 函数

    通俗解释：
    -----------
    输入状态，输出每个动作的 Q 值（价值）

    对于 CartPole：
    - 输入：[位置, 速度, 角度, 角速度] (4 维)
    - 输出：[向左推的价值, 向右推的价值] (2 维)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNPolicyBase(TorchPolicy):
    """
    DQN 策略基类

    使用 ε-贪婪策略：
    - 有 ε 的概率随机选择动作（探索）
    - 有 1-ε 的概率选择 Q 值最大的动作（利用）

    Attributes:
        q_network: Q 网络
        epsilon: 探索率 ε
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        epsilon: float = 0.1,
        device: str = "auto"
    ):
        """
        初始化 DQN 策略

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层大小
            epsilon: 探索率 ε
            device: 计算设备
        """
        super().__init__(device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon

        # 创建 Q 网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_network = self.q_network.to(self.device)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-贪婪策略）

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            动作（整数索引）
        """
        # === 探索阶段 ===
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # === 利用阶段 ===
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(1).item()

        return action

    def set_epsilon(self, epsilon: float):
        """设置探索率"""
        self.epsilon = epsilon

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        获取所有动作的 Q 值（用于调试）

        Args:
            state: 当前状态

        Returns:
            每个动作的 Q 值
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]

    def learn(self, *args, **kwargs) -> dict:
        """
        学习（由外部训练器处理，这里保留接口）

        DQN 的学习需要经验池和优化器，
        我们把它们放在训练器里而不是策略里。
        """
        return {}

    def state_dict(self):
        """获取网络参数"""
        return self.q_network.state_dict()

    def load_state_dict(self, state_dict):
        """加载网络参数"""
        self.q_network.load_state_dict(state_dict)


# 保持向后兼容的别名
DQNPolicy = DQNPolicyBase
