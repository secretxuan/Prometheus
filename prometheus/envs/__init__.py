"""
=============================================
Prometheus 环境模块
=============================================

这个模块处理强化学习环境的抽象和包装。
"""

from prometheus.envs.base import EnvSpec, EnvWrapper
from prometheus.envs.gym_wrapper import GymWrapper, make_gym_env

__all__ = [
    "EnvSpec",
    "EnvWrapper",
    "GymWrapper",
    "make_gym_env",
]
