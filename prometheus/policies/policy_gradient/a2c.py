"""
=============================================
A2C（Actor-Critic）策略
=============================================

什么是 Actor-Critic？
---------------------
Actor-Critic 结合了策略梯度和价值函数方法：

1. **Actor（演员）**：策略网络 π(a|s)，负责选择动作
2. **Critic（评论家）**：价值网络 V(s)，负责评估状态价值

A2C = Advantage Actor-Critic
----------------------------
使用 Advantage 而不是 Return 来更新策略：

Advantage = Q(s,a) - V(s) ≈ r + γV(s') - V(s)

通俗解释：
-----------
- Critic 告诉 Actor：当前状态有多好
- Actor 根据 Critic 的评估调整策略
- Advantage 表示：这个动作比平均好多少

相比 REINFORCE 的优势：
1. 方差更低（因为 Critic 提供了基线）
2. 可以在每个步骤更新（不需要等 episode 结束）
3. 收敛更快
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from prometheus.policies.base import TorchPolicy


class ActorNetwork(nn.Module):
    """
    Actor 网络（策略网络）

    输入状态，输出每个动作的概率分布
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """获取归一化的动作概率"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)

    def get_action_log_probs(self, state: torch.Tensor) -> torch.Tensor:
        """获取动作 log 概率"""
        logits = self.forward(state)
        return F.log_softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """
    Critic 网络（价值网络）

    输入状态，输出状态价值 V(s)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            状态价值 [batch_size, 1]
        """
        return self.network(state)


class A2CPolicy(TorchPolicy):
    """
    A2C 策略

    包含 Actor 和 Critic 两个网络
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "auto"
    ):
        """
        初始化 A2C 策略

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层大小
            gamma: 折扣因子
            entropy_coef: 熵正则化系数（鼓励探索）
            device: 计算设备
        """
        super().__init__(device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # 创建 Actor 和 Critic
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        # 优化器（外部设置）
        self.actor_optimizer = None
        self.critic_optimizer = None

        # 存储一个 episode 的数据
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.actions = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            动作（整数索引）
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 获取动作概率
            probs = self.actor.get_action_probs(state_tensor)
            # 获取状态价值
            value = self.critic(state_tensor)

        # 创建分布并采样
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        if training:
            self.log_probs.append(dist.log_prob(action))
            self.values.append(value)
            self.states.append(state)
            self.actions.append(action.item())

        return action.item()

    def learn(self, next_state: np.ndarray = None, next_done: bool = False) -> dict:
        """
        学习

        Args:
            next_state: 下一个状态（用于计算最后一个状态的 TD 目标）
            next_done: 下一个状态是否结束

        Returns:
            包含损失值等信息的字典
        """
        if not self.log_probs or not self.rewards:
            return {}

        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise ValueError("优化器未设置，请先调用 set_optimizer()")

        # === 计算 Advantages ===
        advantages, returns = self._compute_advantages(next_state, next_done)

        # 转换成张量
        log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).squeeze().to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # === 计算 Actor 损失 ===
        # 策略梯度：-log π(a|s) * A(s,a)
        policy_loss = -(log_probs * advantages).mean()

        # === 计算 Critic 损失 ===
        # 价值函数：MSE( V(s), R )
        value_loss = F.mse_loss(values, returns)

        # === 计算熵损失（鼓励探索）===
        # 熵越大，分布越均匀，探索越多
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy

        # === 总损失 ===
        actor_loss = policy_loss + entropy_loss

        # === 更新 Actor ===
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # === 更新 Critic ===
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # === 清空缓存 ===
        result = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item()
        }
        self._reset_memory()

        return result

    def _compute_advantages(
        self,
        next_state: np.ndarray = None,
        next_done: bool = False
    ) -> Tuple[List[float], List[float]]:
        """
        计算 Advantage 和 Return

        使用 TD 残差作为 Advantage：
        A(s,a) = r + γV(s') - V(s)

        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []

        # 计算下一个状态的价值
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_value = self.critic(next_state_tensor).item()
        else:
            next_value = 0

        # 从后往前计算
        R = 0
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                R = 0
            else:
                R = self.rewards[i] + self.gamma * next_value

            # 最后一轮的特殊处理
            if i == len(self.rewards) - 1 and next_state is not None:
                advantage = R - self.values[i].item()
            elif i < len(self.rewards) - 1:
                next_value_cached = self.values[i + 1].item()
                if self.dones[i + 1]:
                    next_value_cached = 0
                advantage = self.rewards[i] + self.gamma * next_value_cached - self.values[i].item()
                R = self.rewards[i] + self.gamma * next_value_cached
            else:
                advantage = R - self.values[i].item()

            advantages.insert(0, advantage)
            returns.insert(0, R)
            next_value = self.values[i].item()

        return advantages, returns

    def _reset_memory(self):
        """清空经验缓存"""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.actions = []

    def set_optimizer(self, actor_optimizer, critic_optimizer):
        """设置优化器"""
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def store_reward(self, reward: float, done: bool):
        """
        存储奖励和 done 标记

        Args:
            reward: 获得的奖励
            done: 是否结束
        """
        self.rewards.append(reward)
        self.dones.append(done)

    def reset_episode(self):
        """重置 episode 数据"""
        self._reset_memory()

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        获取动作概率（用于调试）

        Args:
            state: 当前状态

        Returns:
            每个动作的概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor.get_action_probs(state_tensor)
        return probs.cpu().numpy()[0]

    def get_state_value(self, state: np.ndarray) -> float:
        """
        获取状态价值（用于调试）

        Args:
            state: 当前状态

        Returns:
            状态价值 V(s)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return value.item()

    def state_dict(self):
        """获取网络参数"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        """加载网络参数"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
