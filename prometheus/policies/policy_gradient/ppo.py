"""
=============================================
PPO（Proximal Policy Optimization）策略
=============================================

什么是 PPO？
------------
PPO 是目前最流行、最实用的强化学习算法之一。

核心思想：限制策略更新的幅度

问题背景：
-----------
策略梯度方法有一个大问题：如果一次更新太大，
新策略可能变得很糟糕，而且很难恢复。

PPO 的解决方案：
----------------
使用 Clipped Surrogate Objective：

L_CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

其中：
- r(θ) = π_new(a|s) / π_old(a|s) 是概率比
- A 是 Advantage
- ε 是裁剪参数（通常是 0.2）

通俗解释：
-----------
想象你在调参：
1. 如果新策略比旧策略好很多（r 很大），我们也给它一个上限
2. 如果新策略比旧策略差很多（r 很小），我们也不惩罚太多
3. 这样可以防止策略"一步走错"

为什么 PPO 这么受欢迎？
-----------------------
1. 简单：容易实现和调试
2. 稳定：不容易崩溃
3. 高效：每次收集的数据可以多次使用
4. 通用：适用于各种任务
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from collections import deque

from prometheus.policies.base import TorchPolicy


class ActorNetwork(nn.Module):
    """
    Actor 网络（策略网络）
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
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
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOPolicy(TorchPolicy):
    """
    PPO 策略

    使用 Clipped Surjective Objective
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "auto"
    ):
        """
        初始化 PPO 策略

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层大小
            gamma: 折扣因子
            gae_lambda: GAE 参数（用于计算 Advantage）
            clip_epsilon: PPO 裁剪参数
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            device: 计算设备
        """
        super().__init__(device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # 创建 Actor 和 Critic
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        # 旧策略（用于 PPO）
        self.old_actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.old_actor = self.old_actor.to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_actor.eval()

        # 优化器
        self.optimizer = None

        # 存储收集的数据
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

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
            # 获取动作概率（使用当前策略）
            probs = self.actor.get_action_probs(state_tensor)
            # 获取状态价值
            value = self.critic(state_tensor)

        # 创建分布并采样
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(dist.log_prob(action))
            self.values.append(value)

        return action.item()

    def remember(self, reward: float, done: bool):
        """
        存储奖励和 done 标记

        Args:
            reward: 获得的奖励
            done: 是否结束
        """
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self, n_epochs: int = 4, batch_size: int = 64) -> dict:
        """
        学习

        PPO 特点：收集一批数据后，多次使用这批数据更新

        Args:
            n_epochs: 使用同一批数据更新多少次
            batch_size: 批量大小

        Returns:
            包含损失值等信息的字典
        """
        if len(self.states) == 0:
            return {}

        if self.optimizer is None:
            raise ValueError("优化器未设置，请先调用 set_optimizer()")

        # === 计算 Advantages 和 Returns ===
        advantages, returns = self._compute_gae()

        # 转换成张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        old_values = torch.stack(self.values).squeeze().to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 归一化 Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === 多次更新 ===
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # 生成随机索引
        indices = np.arange(len(states))

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # === 获取新的 log_probs 和 values ===
                # 使用旧策略计算 log_prob
                with torch.no_grad():
                    old_probs = self.old_actor.get_action_probs(batch_states)
                    old_dist = torch.distributions.Categorical(old_probs)
                    batch_old_log_probs = old_dist.log_prob(batch_actions)

                # 使用当前策略计算 log_prob
                new_probs = self.actor.get_action_probs(batch_states)
                new_dist = torch.distributions.Categorical(new_probs)
                new_log_probs = new_dist.log_prob(batch_actions)

                # 获取新的 values
                new_values = self.critic(batch_states).squeeze()

                # === 计算比率 r(θ) ===
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # === PPO Clipped Objective ===
                # L_CLIP = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # === 价值函数损失 ===
                value_loss = F.mse_loss(new_values, batch_returns)

                # === 熵损失（鼓励探索）===
                entropy = new_dist.entropy().mean()

                # === 总损失 ===
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # === 更新 ===
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # === 更新旧策略 ===
        self.old_actor.load_state_dict(self.actor.state_dict())

        # === 清空缓存 ===
        n_updates = n_epochs * (len(states) // batch_size + 1)
        result = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates
        }
        self._reset_memory()

        return result

    def _compute_gae(self) -> Tuple[List[float], List[float]]:
        """
        使用 GAE（Generalized Advantage Estimation）计算 Advantage

        GAE(λ) = Σ (γλ)^t * δ_t
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []

        # 计算最后一个状态的 value
        with torch.no_grad():
            last_value = 0
            if len(self.states) > 0 and not self.dones[-1]:
                last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
                last_value = self.critic(last_state).item()

        # GAE 计算
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                delta = self.rewards[i] - self.values[i].item()
                gae = delta
            else:
                if i == len(self.rewards) - 1:
                    next_value = last_value
                else:
                    next_value = self.values[i + 1].item()
                delta = self.rewards[i] + self.gamma * next_value - self.values[i].item()
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i].item())

        return advantages, returns

    def _reset_memory(self):
        """清空经验缓存"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def reset_episode(self):
        """
        重置 episode 数据

        注意：PPO 不在 episode 结束时清空，
        而是收集一定数量步骤后再清空
        """
        pass

    def set_optimizer(self, optimizer):
        """设置优化器"""
        self.optimizer = optimizer

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
            'critic': self.critic.state_dict(),
            'old_actor': self.old_actor.state_dict()
        }

    def load_state_dict(self, state_dict):
        """加载网络参数"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.old_actor.load_state_dict(state_dict['old_actor'])
