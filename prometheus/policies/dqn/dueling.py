"""
=============================================
Dueling DQN 策略
=============================================

Dueling DQN 将 Q 值分解为状态价值和动作优势：

    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

其中：
- V(s): 状态价值，表示这个状态本身有多好
- A(s,a): 动作优势，表示在这个状态下，某个动作比平均动作好多少

优势：
1. 对于某些状态，所有动作的价值都相近（只需要学习 V(s)）
2. 更好地处理动作无关的状态评估

论文: Dueling Network Architectures for Deep Reinforcement Learning (2016)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from prometheus.policies.base import TorchPolicy


class DuelingQNetwork(nn.Module):
    """
    Dueling Q 网络

    结构：
                    共享特征提取层
                         |
          +--------------+--------------+
          |                             |
    Value Stream                  Advantage Stream
    (状态价值 V)                   (动作优势 A)
          |                             |
          +--------------+--------------+
                         |
                   Q(s,a) = V + A - mean(A)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # === 共享特征提取层 ===
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # === Value Stream (状态价值) ===
        # 输出标量 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # === Advantage Stream (动作优势) ===
        # 输出每个动作的优势 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

        这个公式保证了：
        1. 当优势为 0 时，Q 值等于状态价值
        2. 增加某个动作的优势，不会影响其他动作的 Q 值排序
        """
        # 共享特征
        features = self.shared_layer(state)

        # 状态价值 V(s)
        value = self.value_stream(features)  # [batch, 1]

        # 动作优势 A(s,a)
        advantage = self.advantage_stream(features)  # [batch, action_dim]

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        # 减去优势的均值是为了保持可识别性
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class DuelingDQNPolicy(TorchPolicy):
    """
    Dueling DQN 策略

    使用 DuelingQNetwork 代替标准 QNetwork
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        epsilon: float = 0.1,
        device: str = "auto"
    ):
        super().__init__(device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon

        # 创建 Dueling Q 网络
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim)
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
        import random

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

    def get_value_and_advantage(self, state: np.ndarray):
        """
        分别获取状态价值和动作优势（用于调试和分析）

        Args:
            state: 当前状态

        Returns:
            (value, advantage) 元组
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            features = self.q_network.shared_layer(state_tensor)
            value = self.q_network.value_stream(features).cpu().numpy()[0, 0]
            advantage = self.q_network.advantage_stream(features).cpu().numpy()[0]
        return value, advantage

    def learn(self, *args, **kwargs) -> dict:
        """学习接口"""
        return {}

    def state_dict(self):
        """获取网络参数"""
        return self.q_network.state_dict()

    def load_state_dict(self, state_dict):
        """加载网络参数"""
        self.q_network.load_state_dict(state_dict)
