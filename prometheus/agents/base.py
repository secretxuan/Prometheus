"""
=============================================
智能体抽象基类
=============================================

什么是智能体（Agent）？
-----------------------
智能体是强化学习中的"决策者"或"玩家"。

通俗解释：
-----------
智能体就像一个玩游戏的玩家：
1. 观察游戏状态
2. 决定做什么动作
3. 从环境中获得奖励
4. 根据奖励调整策略，越玩越好

智能体的组成：
-------------
- 策略（Policy）：如何选择动作
- 学习算法：如何从经验中学习
- 记忆（可选）：存储过去的经验
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseAgent(ABC):
    """
    智能体抽象基类

    所有智能体都必须实现这个接口
    """

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Any:
        """
        选择动作

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            动作
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs) -> Dict[str, float]:
        """
        学习/更新

        Args:
            *args: 学习所需的输入
            **kwargs: 额外参数

        Returns:
            学习信息字典（如损失值、指标等）
        """
        pass

    @abstractmethod
    def remember(self, *args):
        """
        存储经验

        Args:
            *args: 要存储的经验数据
        """
        pass

    @abstractmethod
    def reset(self):
        """重置智能体状态（新 episode 开始时）"""
        pass

    def save(self, path: str):
        """
        保存智能体

        Args:
            path: 保存路径
        """
        raise NotImplementedError

    def load(self, path: str):
        """
        加载智能体

        Args:
            path: 模型路径
        """
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, training: bool = True):
        """
        设置训练/评估模式

        Args:
            training: True=训练模式, False=评估模式
        """
        pass
