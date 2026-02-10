"""
=============================================
Rainbow DQN 智能体
=============================================

Rainbow 整合了 DQN 的多种改进：
- Dueling DQN 网络结构
- Double DQN 的目标 Q 计算
- 优先级经验回放（PER）

论文: Rainbow: Combining Improvements in Deep Reinforcement Learning (2017)
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict

from prometheus.agents.base import BaseAgent
from prometheus.policies.dqn.rainbow import RainbowDQNPolicy
from prometheus.policies.dqn.dueling import DuelingQNetwork
from prometheus.core import PrioritizedReplayBuffer, Config


class RainbowAgent(BaseAgent):
    """
    Rainbow DQN 智能体

    整合的改进：
    1. Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)
    2. Double DQN: 主网络选动作，目标网络评估
    3. PER: 按优先级采样，用重要性采样权重修正

    通俗解释：
    -----------
    Rainbow 就像是 DQN 的"豪华升级版"：
    - 用更聪明的网络结构（Dueling）
    - 更准确的计算方式（Double）
    - 更高效的学习策略（PER）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config = None,
        device: str = "auto",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        初始化 Rainbow 智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            config: 配置对象
            device: 计算设备
            alpha: PER 优先级指数
            beta_start: PER beta 初始值
            beta_frames: PER beta 增长步数
        """
        if config is None:
            config = Config()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # === 创建策略（使用 DuelingQNetwork）===
        self.policy = RainbowDQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            epsilon=config.EPSILON_START,
            device=device
        )

        # === 创建目标网络（同样使用 DuelingQNetwork）===
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim=128)
        self.target_network = self.target_network.to(self.policy.device)
        self.target_network.load_state_dict(self.policy.q_network.state_dict())

        # === 创建优化器 ===
        self.optimizer = optim.Adam(
            self.policy.q_network.parameters(),
            lr=config.LEARNING_RATE
        )

        # === 创建优先级经验回放缓冲区 ===
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.MEMORY_SIZE,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames
        )

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

        整合了 Double DQN 和 PER 的学习方式
        """
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return {}

        # === 采样经验（返回索引和重要性采样权重）===
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.config.BATCH_SIZE)

        # 转换成张量
        states = torch.FloatTensor(states).to(self.policy.device)
        actions = torch.LongTensor(actions).to(self.policy.device)
        rewards = torch.FloatTensor(rewards).to(self.policy.device)
        next_states = torch.FloatTensor(next_states).to(self.policy.device)
        dones = torch.FloatTensor(dones).to(self.policy.device)
        weights = torch.FloatTensor(weights).to(self.policy.device)

        # === Double DQN: 解耦动作选择和价值评估 ===
        with torch.no_grad():
            # 1. 用主网络选择下一个动作
            next_q_policy = self.policy.q_network(next_states)
            next_actions = next_q_policy.argmax(1)

            # 2. 用目标网络评估该动作的价值
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # 3. 计算 target Q
            target_q = rewards + self.config.GAMMA * next_q_values * (1 - dones)

        # === 计算当前 Q 值 ===
        current_q = self.policy.q_network(states)
        q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # === 计算 TD 误差（用于更新优先级）===
        td_errors = torch.abs(target_q - q_values)

        # === 计算加权损失 ===
        loss = (weights * torch.nn.SmoothL1Loss(reduction='none')(q_values, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # === 更新优先级 ===
        self.replay_buffer.update_priorities(indices, td_errors.cpu().detach().numpy())

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
RainbowDQN = RainbowAgent
