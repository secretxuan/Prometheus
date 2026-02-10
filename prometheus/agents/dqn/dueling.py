"""
=============================================
Dueling DQN 智能体
=============================================
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any

from prometheus.agents.base import BaseAgent
from prometheus.policies.dqn.dueling import DuelingDQNPolicy, DuelingQNetwork
from prometheus.core import ReplayBuffer, Config


class DuelingDQNAgent(BaseAgent):
    """
    Dueling DQN 智能体

    使用 DuelingQNetwork，将 Q 值分解为 V(s) + A(s,a)

    通俗解释：
    -----------
    标准 DQN 直接学习每个动作的 Q 值。

    Dueling DQN 将 Q 值拆成两部分：
    - V(s): 这个状态本身有多好（比如"快到终点了"就是好状态）
    - A(s,a): 这个动作比平均动作好多少

    这样做的好处是，某些状态下所有动作都差不多好，
    智能体只需要学习"这是个好状态"，而不需要精确区分每个动作。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config = None,
        device: str = "auto"
    ):
        """
        初始化 Dueling DQN 智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            config: 配置对象
            device: 计算设备
        """
        if config is None:
            config = Config()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # === 创建策略（使用 DuelingQNetwork）===
        self.policy = DuelingDQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            epsilon=config.EPSILON_START,
            device=device
        )

        # === 创建目标网络 ===
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim=128)
        self.target_network = self.target_network.to(self.policy.device)
        self.target_network.load_state_dict(self.policy.q_network.state_dict())

        # === 创建优化器 ===
        self.optimizer = optim.Adam(
            self.policy.q_network.parameters(),
            lr=config.LEARNING_RATE
        )

        # === 创建经验池 ===
        self.replay_buffer = ReplayBuffer(config.MEMORY_SIZE)

        # === 训练状态 ===
        self.training = True
        self._epsilon = config.EPSILON_START

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        return self.policy.select_action(state, training=training)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Dict[str, float]:
        """
        学习（训练一次）

        Dueling DQN 的学习过程与标准 DQN 相同，
        差异只在网络结构上。
        """
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return {}

        # === 采样经验 ===
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.BATCH_SIZE
        )

        # 转换成张量
        states = torch.FloatTensor(states).to(self.policy.device)
        actions = torch.LongTensor(actions).to(self.policy.device)
        rewards = torch.FloatTensor(rewards).to(self.policy.device)
        next_states = torch.FloatTensor(next_states).to(self.policy.device)
        dones = torch.FloatTensor(dones).to(self.policy.device)

        # === 计算目标 Q 值 ===
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_max = next_q_values.max(1)[0]
            target_q = rewards + self.config.GAMMA * next_q_max * (1 - dones)

        # === 计算当前 Q 值 ===
        current_q = self.policy.q_network(states)
        q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # === 计算损失并更新 ===
        loss = torch.nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # === 更新探索率 ===
        self._epsilon = max(
            self.config.EPSILON_END,
            self._epsilon * self.config.EPSILON_DECAY
        )
        self.policy.set_epsilon(self._epsilon)

        return {"loss": loss.item(), "epsilon": self._epsilon}

    def update_target_network(self):
        """同步目标网络"""
        self.target_network.load_state_dict(self.policy.q_network.state_dict())

    def reset(self):
        """重置"""
        pass

    def set_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.training = training
        self.policy.set_mode(training)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.policy.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self._epsilon
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self._epsilon = checkpoint['epsilon']
        self.policy.set_epsilon(self._epsilon)


# 保持向后兼容的别名
DuelingDQN = DuelingDQNAgent
