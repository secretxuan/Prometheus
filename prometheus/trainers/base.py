"""
=============================================
训练器抽象基类
=============================================

什么是训练器（Trainer）？
------------------------
训练器负责"管理整个训练过程"。

通俗解释：
-----------
训练器就像"教练"：
1. 安排训练计划
2. 管理环境和智能体
3. 记录训练过程
4. 评估学习效果

训练器 vs 智能体：
-----------------
- 智能体：学习如何玩游戏
- 训练器：安排训练，记录进度
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
from pathlib import Path


@dataclass
class TrainerConfig:
    """
    训练器配置

    通俗解释：
    -----------
    训练器的"设置面板"，控制训练过程的各种参数
    """

    max_episodes: int = 1000          # 最大训练轮数
    max_steps_per_episode: int = 1000  # 每轮最大步数

    # 评估相关
    eval_episodes: int = 10           # 评估时玩几轮
    eval_interval: int = 100          # 每隔多少轮评估一次

    # 保存相关
    save_interval: int = 500          # 每隔多少轮保存一次
    save_dir: str = "checkpoints"     # 保存目录

    # 日志相关
    log_interval: int = 10            # 每隔多少轮打印一次
    use_tensorboard: bool = False     # 是否使用 TensorBoard

    # 回调函数
    callbacks: List[Callable] = field(default_factory=list)


class BaseTrainer(ABC):
    """
    训练器抽象基类

    所有训练器都必须实现这个接口
    """

    def __init__(self, config: TrainerConfig = None):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config or TrainerConfig()
        self._current_episode = 0

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        开始训练

        Args:
            **kwargs: 额外参数

        Returns:
            训练结果字典
        """
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, float]:
        """
        评估智能体

        Args:
            **kwargs: 额外参数

        Returns:
            评估结果字典
        """
        pass

    def save_checkpoint(self, path: str):
        """
        保存检查点

        Args:
            path: 保存路径
        """
        raise NotImplementedError

    def load_checkpoint(self, path: str):
        """
        加载检查点

        Args:
            path: 模型路径
        """
        raise NotImplementedError


class Callback:
    """
    回调函数基类

    通俗解释：
    -----------
    回调就像训练过程中的"钩子"，在特定时刻执行自定义代码。

    常见用途：
    - 记录训练曲线
    - 保存模型
    - 打印进度
    - 早停（当不再进步时停止训练）
    """

    def on_train_start(self, trainer):
        """训练开始时"""
        pass

    def on_train_end(self, trainer):
        """训练结束时"""
        pass

    def on_episode_start(self, trainer, episode: int):
        """每个 episode 开始时"""
        pass

    def on_episode_end(self, trainer, episode: int, metrics: Dict):
        """每个 episode 结束时"""
        pass

    def on_step(self, trainer, step: int, **kwargs):
        """每一步"""
        pass
