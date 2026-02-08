"""
=============================================
策略抽象基类
=============================================

什么是策略（Policy）？
----------------------
策略就是"根据观察决定做什么动作"的规则。

数学表示：π(a|s) = 在状态 s 下选择动作 a 的概率

通俗解释：
-----------
- 策略就像一本"行动指南"
- 输入：当前看到了什么（状态）
- 输出：应该做什么（动作）

例子：
- 贪婪策略：选择价值最高的动作
- 随机策略：完全随机选择
- ε-贪婪策略：大部分时候选最好的，偶尔随机
"""

from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import torch


class BasePolicy(ABC):
    """
    策略抽象基类

    所有策略都必须实现这个接口
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, **kwargs) -> Any:
        """
        根据状态选择动作

        Args:
            state: 当前状态（观察）
            **kwargs: 额外参数，如 training=True/False

        Returns:
            选择的动作
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs) -> dict:
        """
        学习/更新策略

        Args:
            *args: 学习所需的输入
            **kwargs: 额外参数

        Returns:
            学习信息的字典（如损失值等）
        """
        pass

    def set_mode(self, training: bool = True):
        """
        设置训练/评估模式

        Args:
            training: True=训练模式, False=评估模式
        """
        pass


class StatefulMixin:
    """
    有状态的策略混入类

    通俗解释：
    -----------
    有些策略需要"记住"之前的状态（比如 RNN），
    这个类提供相关功能。
    """

    def reset_state(self):
        """重置内部状态（新 episode 开始时调用）"""
        pass

    def get_state(self):
        """获取当前内部状态"""
        return None

    def set_state(self, state):
        """设置内部状态"""
        pass


class TorchPolicy(BasePolicy):
    """
    基于 PyTorch 的策略基类

    通俗解释：
    -----------
    如果策略使用神经网络，这个类提供了一些通用功能：
    - 自动管理设备（CPU/GPU）
    - 训练/评估模式切换
    """

    def __init__(self, device: str = "auto"):
        """
        初始化

        Args:
            device: "auto" 自动选择, "cpu", "cuda"
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.training = True

    def to(self, device: Union[str, torch.device]):
        """移动模型到指定设备"""
        self.device = torch.device(device)
        return self

    def set_mode(self, training: bool = True):
        """
        设置训练/评估模式

        Args:
            training: True=训练模式, False=评估模式

        注意：
        -----
        训练模式下，可能使用探索策略
        评估模式下，只使用最优策略
        """
        self.training = training
