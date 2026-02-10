"""
=============================================
Rainbow DQN 策略
=============================================

Rainbow 整合了 DQN 的多种改进：
- Dueling DQN 网络结构（V(s) + A(s,a)）
- 可以配合 Double DQN 和 PER 使用

论文: Rainbow: Combining Improvements in Deep Reinforcement Learning (2017)
"""

import torch
import torch.nn as nn
import numpy as np

from prometheus.policies.base import TorchPolicy
from prometheus.policies.dqn.dueling import DuelingQNetwork


class RainbowDQNPolicy(TorchPolicy):
    """
    Rainbow DQN 策略

    使用 DuelingQNetwork 作为基础网络结构，
    可以配合 Double DQN 和 PER 使用。
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

        # 使用 DuelingQNetwork（Rainbow 的核心改进之一）
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim)
        self.q_network = self.q_network.to(self.device)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作（ε-贪婪策略）"""
        import random

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(1).item()

        return action

    def set_epsilon(self, epsilon: float):
        """设置探索率"""
        self.epsilon = epsilon

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取所有动作的 Q 值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]

    def get_value_and_advantage(self, state: np.ndarray):
        """分别获取状态价值和动作优势"""
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
