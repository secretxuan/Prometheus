"""
DQN 策略模块
"""

from prometheus.policies.dqn.base import DQNPolicyBase, DQNPolicy, QNetwork
from prometheus.policies.dqn.double import DoubleDQNPolicy
from prometheus.policies.dqn.dueling import DuelingDQNPolicy, DuelingQNetwork
from prometheus.policies.dqn.rainbow import RainbowDQNPolicy

__all__ = [
    "DQNPolicyBase",
    "DQNPolicy",
    "QNetwork",
    "DoubleDQNPolicy",
    "DuelingDQNPolicy",
    "DuelingQNetwork",
    "RainbowDQNPolicy",
]
