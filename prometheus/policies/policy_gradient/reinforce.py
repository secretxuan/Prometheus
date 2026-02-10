"""
=============================================
REINFORCE 策略
=============================================

什么是 REINFORCE？
------------------
REINFORCE（蒙特卡洛策略梯度）是最基础的策略梯度算法。

核心思想：
1. 直接参数化策略 π(a|s; θ)
2. 如果某个动作获得了高回报，就增加该动作被选择的概率
3. 如果获得了低回报，就降低该动作被选择的概率

数学原理：
-----------
目标：最大化期望回报

∇θ J(θ) = E[∇θ log π(a|s; θ) * G(t)]

其中：
- G(t) 是从时刻 t 开始的回报（return）
- ∇θ log π(a|s; θ) 是策略梯度的对数导数
- 乘积的意思是：如果回报高，增加该动作的概率

通俗解释：
-----------
想象你在玩游戏：
1. 你做出一系列动作
2. 游戏结束后，你看到最终得分
3. 如果得分高，就"记住"刚才做的动作，下次更可能做
4. 如果得分低，就"忘记"刚才做的动作，下次少做

REINFORCE 就是这个想法的数学实现。
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from prometheus.policies.base import TorchPolicy


class PolicyNetwork(nn.Module):
    """
    策略网络

    输入状态，输出每个动作的概率分布

    对于 CartPole：
    - 输入：[位置, 速度, 角度, 角速度] (4 维)
    - 输出：[向左推的概率, 向右推的概率] (2 维，Softmax 后)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            动作 logits [batch_size, action_dim]
        """
        return self.network(state)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取动作概率分布

        Args:
            state: 状态张量

        Returns:
            归一化的动作概率
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class REINFORCEPolicy(TorchPolicy):
    """
    REINFORCE 策略

    使用策略梯度方法直接学习策略

    特点：
    - On-policy：只能用当前策略收集的数据学习
    - 蒙特卡洛：需要完整 episode 才能更新
    - 高方差：单次估计可能很不稳定
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        device: str = "auto"
    ):
        """
        初始化 REINFORCE 策略

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层大小
            gamma: 折扣因子
            device: 计算设备
        """
        super().__init__(device=device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # 创建策略网络
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_network = self.policy_network.to(self.device)

        # 优化器
        self.optimizer = None  # 由外部设置

        # 存储一个 episode 的数据
        self.saved_log_probs = []
        self.saved_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（采样自策略分布）

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            动作（整数索引）
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 获取动作概率
        with torch.no_grad():
            probs = self.policy_network.get_action_probs(state_tensor)

        # 创建分布并采样
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # 训练模式下保存 log_prob（用于学习）
        if training:
            self.saved_log_probs.append(dist.log_prob(action))

        return action.item()

    def learn(self, *args, **kwargs) -> dict:
        """
        学习（基于当前 episode 的经验）

        Returns:
            包含损失值等信息的字典
        """
        if not self.saved_log_probs or not self.saved_rewards:
            return {}

        if self.optimizer is None:
            raise ValueError("优化器未设置，请先调用 set_optimizer()")

        # === 计算折扣回报 ===
        returns = self._compute_returns()

        # 归一化回报（稳定训练）
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # === 计算策略梯度损失 ===
        # 损失 = -log π(a|s) * G
        # 注意：我们最小化负值相当于最大化
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # === 更新网络 ===
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # === 清空缓存 ===
        loss_value = policy_loss.item()
        self.saved_log_probs = []
        self.saved_rewards = []

        return {"loss": loss_value, "mean_return": returns.mean().item()}

    def _compute_returns(self) -> list:
        """
        计算折扣回报

        G(t) = r(t) + γ * r(t+1) + γ² * r(t+2) + ...

        Returns:
            每个时间步的回报列表
        """
        returns = []
        R = 0
        # 从后往前计算
        for reward in reversed(self.saved_rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def set_optimizer(self, optimizer):
        """设置优化器"""
        self.optimizer = optimizer

    def store_reward(self, reward: float):
        """
        存储奖励（在每个步骤后调用）

        Args:
            reward: 获得的奖励
        """
        self.saved_rewards.append(reward)

    def reset_episode(self):
        """重置 episode 数据"""
        self.saved_log_probs = []
        self.saved_rewards = []

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
            probs = self.policy_network.get_action_probs(state_tensor)
        return probs.cpu().numpy()[0]

    def state_dict(self):
        """获取网络参数"""
        return self.policy_network.state_dict()

    def load_state_dict(self, state_dict):
        """加载网络参数"""
        self.policy_network.load_state_dict(state_dict)
