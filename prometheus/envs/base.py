"""
=============================================
环境抽象基类
=============================================

这个文件定义了环境的抽象接口。

什么是环境抽象？
----------------
不同的 RL 库有不同的环境接口（Gym、Ray、PettingZoo...）
我们需要一个统一的方式来处理它们。

这里定义的抽象类就像"插座适配器"，让不同的环境都能接入我们的框架。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Optional
import numpy as np


# ============================================================
# 环境规格描述
# ============================================================

@dataclass
class EnvSpec:
    """
    环境规格 - 描述环境的基本信息

    通俗解释：
    ---------
    就像产品的"说明书"，告诉你这个环境：
    - 观察空间是什么样子的
    - 动作空间是什么样子的
    - 最大步数是多少
    """

    name: str                          # 环境名称，如 "CartPole-v1"
    observation_space: "Space"         # 观察空间（状态空间）
    action_space: "Space"              # 动作空间
    max_episode_steps: Optional[int]   # 最大步数


# ============================================================
# 空间抽象
# ============================================================

class Space(ABC):
    """
    空间抽象基类

    通俗解释：
    ---------
    "空间"就是"所有可能取值的集合"

    例子：
    - 离散空间(Discrete)：{0, 1, 2}，比如 3 个动作
    - 连续空间(Box)：[-1, 1] × [-1, 1]，比如坐标
    """

    @abstractmethod
    def sample(self) -> Any:
        """
        随机采样一个值

        Returns:
            该空间中的一个随机值
        """
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """
        判断 x 是否在这个空间里

        Args:
            x: 要检查的值

        Returns:
            True 如果 x 在空间内，False 否则
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """返回空间的形状"""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """返回空间的数据类型"""
        pass


# ============================================================
# 环境包装器基类
# ============================================================

class EnvWrapper(ABC):
    """
    环境包装器抽象基类

    通俗解释：
    ---------
    包装器就像"手机壳"，给环境增加额外的功能，
    但不改变环境本身。

    常见用途：
    - 记录奖励和动作
    - 裁剪观察值
    - 堆叠连续几帧图像
    """

    def __init__(self, env: Any):
        """
        初始化包装器

        Args:
            env: 被包装的环境
        """
        self._env = env

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        重置环境

        Args:
            seed: 随机种子

        Returns:
            (初始观察, 信息字典)
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步

        Args:
            action: 要执行的动作

        Returns:
            (观察, 奖励, terminated, truncated, 信息)
        """
        pass

    @property
    @abstractmethod
    def spec(self) -> EnvSpec:
        """获取环境规格"""
        pass

    def close(self):
        """关闭环境"""
        if hasattr(self._env, 'close'):
            self._env.close()


# ============================================================
# Gym 兼容层
# ============================================================

class GymSpace(Space):
    """
    Gym 空间的包装器

    通俗解释：
    ---------
    把 Gym 的空间包装成我们的 Space 接口
    """

    def __init__(self, gym_space):
        self._gym_space = gym_space

    def sample(self) -> Any:
        return self._gym_space.sample()

    def contains(self, x: Any) -> bool:
        return self._gym_space.contains(x)

    @property
    def shape(self) -> Tuple[int, ...]:
        if hasattr(self._gym_space, 'shape'):
            return self._gym_space.shape
        return ()

    @property
    def dtype(self) -> np.dtype:
        if hasattr(self._gym_space, 'dtype'):
            return self._gym_space.dtype
        return np.float64
