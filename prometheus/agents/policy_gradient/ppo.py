"""
=============================================
PPO 智能体
=============================================
"""

import torch.optim as optim
import numpy as np
import torch
from typing import Dict, Any

from prometheus.agents.base import BaseAgent
from prometheus.policies.policy_gradient.ppo import PPOPolicy
from prometheus.core import Config


class PPOAgent(BaseAgent):
    """
    PPO（Proximal Policy Optimization）智能体

    使用 Clipped Surrogate Objective 防止策略更新过大

    特点：
    - Actor-Critic 架构
    - 使用 GAE 计算 Advantage
    - 使用 clipped objective 限制策略更新
    - 一批数据可以多次使用
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "auto"
    ):
        """
        初始化 PPO 智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = Config  # 使用静态 Config 类

        # === 创建策略 ===
        self.policy = PPOPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            gamma=Config.GAMMA,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            device=device
        )

        # === 创建优化器 ===
        self.optimizer = optim.Adam(
            list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()),
            lr=Config.LEARNING_RATE
        )
        self.policy.set_optimizer(self.optimizer)

        # === 训练状态 ===
        self.training = True

        # PPO 特有：收集一定数量的步骤后更新
        self.collect_steps = 2048  # 收集多少步后更新
        self.step_count = 0

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
        if self.training:
            self.policy.remember(reward, done)
            self.step_count += 1

    def learn(self, **kwargs) -> Dict[str, float]:
        """
        学习

        PPO 特点：收集足够数据后进行多次更新

        Returns:
            包含损失值等信息的字典
        """
        return self.policy.learn(
            n_epochs=kwargs.get('n_epochs', 4),
            batch_size=kwargs.get('batch_size', 64)
        )

    def should_update(self) -> bool:
        """是否应该更新（收集了足够数据）"""
        return self.step_count >= self.collect_steps

    def reset(self):
        """重置（新 episode 开始时）"""
        # PPO 不需要重置，因为数据是跨 episode 收集的
        pass

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
            'actor': self.policy.actor.state_dict(),
            'critic': self.policy.critic.state_dict(),
            'old_actor': self.policy.old_actor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.actor.load_state_dict(checkpoint['actor'])
        self.policy.critic.load_state_dict(checkpoint['critic'])
        self.policy.old_actor.load_state_dict(checkpoint['old_actor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
