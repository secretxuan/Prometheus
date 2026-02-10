"""
=============================================
Prometheus 训练器模块
=============================================

训练器（Trainer）负责管理整个训练过程。
"""

from prometheus.trainers.base import BaseTrainer, TrainerConfig
from prometheus.trainers.dqn import DQNTrainer
from prometheus.trainers.policy_gradient import (
    REINFORCETrainer,
    A2CTrainer,
    PPOTrainer
)

__all__ = [
    "BaseTrainer",
    "TrainerConfig",
    "DQNTrainer",
    "REINFORCETrainer",
    "A2CTrainer",
    "PPOTrainer",
]
