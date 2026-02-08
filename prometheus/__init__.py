"""
=============================================
Prometheus - 强化学习框架
=============================================

为人类带来火种的 RL 框架

版本: 0.1.0 (铸炉版本)

模块概览：
---------
- prometheus.core: 核心类（Config, ReplayBuffer）
- prometheus.envs: 环境相关（EnvWrapper, make_gym_env）
- prometheus.policies: 策略相关（DQNPolicy）
- prometheus.agents: 智能体相关（DQNAgent）
- prometheus.trainers: 训练器相关（DQNTrainer）

快速开始：
---------
>>> from prometheus.envs import make_gym_env
>>> from prometheus.agents import DQNAgent
>>> from prometheus.trainers import DQNTrainer
>>>
>>> env = make_gym_env("CartPole-v1")
>>> agent = DQNAgent(state_dim=4, action_dim=2)
>>> trainer = DQNTrainer()
>>> trainer.train(env, agent)
"""

__version__ = "0.1.0"
__author__ = "Prometheus Team"

# 导入核心类
from prometheus.core import Config, ReplayBuffer

# 导入环境相关
from prometheus.envs import (
    EnvSpec,
    EnvWrapper,
    GymWrapper,
    make_gym_env
)

# 导入策略相关
from prometheus.policies import (
    BasePolicy,
    DQNPolicy
)

# 导入智能体相关
from prometheus.agents import (
    BaseAgent,
    DQNAgent
)

# 导入训练器相关
from prometheus.trainers import (
    BaseTrainer,
    TrainerConfig,
    DQNTrainer
)

__all__ = [
    # 版本信息
    "__version__",

    # 核心
    "Config",
    "ReplayBuffer",

    # 环境
    "EnvSpec",
    "EnvWrapper",
    "GymWrapper",
    "make_gym_env",

    # 策略
    "BasePolicy",
    "DQNPolicy",

    # 智能体
    "BaseAgent",
    "DQNAgent",

    # 训练器
    "BaseTrainer",
    "TrainerConfig",
    "DQNTrainer",
]
