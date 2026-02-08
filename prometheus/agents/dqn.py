"""
=============================================
DQN 智能体
=============================================

DQN 智能体使用 DQN 策略和经验回放来学习。
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple

from prometheus.agents.base import BaseAgent
from prometheus.policies.dqn import DQNPolicy, QNetwork
from prometheus.core import ReplayBuffer, Config


class DQNAgent(BaseAgent):
    """
    DQN 智能体

    组成部分：
    1. Q 网络（策略网络）：估计每个动作的价值
    2. 目标网络：稳定的训练目标
    3. 经验池：存储和回放经验
    4. 优化器：更新网络参数

    通俗解释：
    -----------
    就像一个学玩游戏的人：
    - 有一个"大脑"（Q 网络）做决策
    - 记住了过去的经历（经验池）
    - 通过回顾经历来学习（训练）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config = None,
        device: str = "auto"
    ):
        """
        初始化 DQN 智能体

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

        # === 创建策略（包含 Q 网络）===
        self.policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            epsilon=config.EPSILON_START,
            device=device
        )

        # === 创建目标网络 ===
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim=128)
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
        """
        选择动作

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            动作
        """
        return self.policy.select_action(state, training=training)

    def remember(self, state, action, reward, next_state, done):
        """
        存储经验

        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Dict[str, float]:
        """
        学习（训练一次）

        Returns:
            包含损失值等信息的字典
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
        """重置（新 episode 开始时）"""
        pass  # DQN 不需要跨 episode 的状态

    def set_mode(self, training: bool = True):
        """
        设置训练/评估模式

        Args:
            training: True=训练模式, False=评估模式
        """
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
