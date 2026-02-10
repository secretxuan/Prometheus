"""
=============================================
REINFORCE 智能体
=============================================
"""

import torch.optim as optim
import numpy as np
from typing import Dict, Any

from prometheus.agents.base import BaseAgent
from prometheus.policies.policy_gradient.reinforce import REINFORCEPolicy
from prometheus.core import Config


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE 智能体

    使用蒙特卡洛策略梯度方法学习

    特点：
    - On-policy：只能用当前策略收集的数据学习
    - 每个 episode 结束后更新一次
    - 不需要经验池
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config = None,
        device: str = "auto"
    ):
        """
        初始化 REINFORCE 智能体

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
        self.policy = REINFORCEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            gamma=config.GAMMA,
            device=device
        )

        # === 创建优化器 ===
        self.optimizer = optim.Adam(
            self.policy.policy_network.parameters(),
            lr=config.LEARNING_RATE
        )
        self.policy.set_optimizer(self.optimizer)

        # === 训练状态 ===
        self.training = True

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

        对于 REINFORCE，只需要存储奖励
        其他信息在 select_action 时已存储

        Args:
            state: 当前状态（不使用）
            action: 动作（不使用，已在 select_action 中记录）
            reward: 奖励
            next_state: 下一状态（不使用）
            done: 是否结束（不使用）
        """
        if self.training:
            self.policy.store_reward(reward)

    def learn(self) -> Dict[str, float]:
        """
        学习（在 episode 结束时调用）

        Returns:
            包含损失值等信息的字典
        """
        return self.policy.learn()

    def reset(self):
        """重置（新 episode 开始时）"""
        self.policy.reset_episode()

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
            'policy_network': self.policy.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
