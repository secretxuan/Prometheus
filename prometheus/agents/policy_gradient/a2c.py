"""
=============================================
A2C 智能体
=============================================
"""

import torch.optim as optim
import numpy as np
from typing import Dict, Any

from prometheus.agents.base import BaseAgent
from prometheus.policies.policy_gradient.a2c import A2CPolicy
from prometheus.core import Config


class A2CAgent(BaseAgent):
    """
    A2C（Advantage Actor-Critic）智能体

    结合策略梯度和价值函数方法

    特点：
    - Actor-Critic 架构
    - 使用 Advantage 减少方差
    - 可以批量更新或在线更新
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config = None,
        device: str = "auto"
    ):
        """
        初始化 A2C 智能体

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

        # === 创建策略 ===
        self.policy = A2CPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            gamma=config.GAMMA,
            entropy_coef=0.01,
            device=device
        )

        # === 创建优化器 ===
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(),
            lr=config.LEARNING_RATE
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(),
            lr=config.LEARNING_RATE
        )
        self.policy.set_optimizer(self.actor_optimizer, self.critic_optimizer)

        # === 训练状态 ===
        self.training = True

        # 用于 n-step 更新
        self.n_steps = 5  # n-step 学习
        self.step_count = 0

        # 当前 episode 的最后一个状态和 done
        self.last_state = None
        self.last_done = False

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
            self.policy.store_reward(reward, done)
            self.last_state = next_state
            self.last_done = done
            self.step_count += 1

    def learn(self, **kwargs) -> Dict[str, float]:
        """
        学习

        可以在每个 episode 结束时学习，或每 n 步学习一次

        Returns:
            包含损失值等信息的字典
        """
        return self.policy.learn(
            next_state=self.last_state,
            next_done=self.last_done
        )

    def should_update(self) -> bool:
        """是否应该更新（用于 n-step 学习）"""
        return self.step_count >= self.n_steps

    def reset(self):
        """重置（新 episode 开始时）"""
        self.policy.reset_episode()
        self.step_count = 0
        self.last_state = None
        self.last_done = False

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
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.actor.load_state_dict(checkpoint['actor'])
        self.policy.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
