"""
DQN 智能体模块
"""

from prometheus.agents.dqn.base import DQNAgentBase, DQNAgent
from prometheus.agents.dqn.double import DoubleDQNAgent, DoubleDQN
from prometheus.agents.dqn.dueling import DuelingDQNAgent, DuelingDQN
from prometheus.agents.dqn.per import PERAgent, PERDQN, PrioritizedReplayDQN
from prometheus.agents.dqn.rainbow import RainbowAgent, RainbowDQN

__all__ = [
    "DQNAgentBase",
    "DQNAgent",
    "DoubleDQNAgent",
    "DoubleDQN",
    "DuelingDQNAgent",
    "DuelingDQN",
    "PERAgent",
    "PERDQN",
    "PrioritizedReplayDQN",
    "RainbowAgent",
    "RainbowDQN",
]
