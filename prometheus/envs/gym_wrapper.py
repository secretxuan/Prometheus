"""
=============================================
Gym 环境包装器
=============================================

这个文件让 Gymnasium 环境能接入我们的框架。
"""

import numpy as np
from typing import Tuple, Optional, Any
from prometheus.envs.base import EnvWrapper, EnvSpec, GymSpace


class GymWrapper(EnvWrapper):
    """
    Gymnasium 环境包装器

    通俗解释：
    ---------
    把 Gymnasium 的环境包装成我们框架的统一接口

    例子：
    -------
    >>> import gymnasium as gym
    >>> env = gym.make("CartPole-v1")
    >>> wrapped_env = GymWrapper(env)
    >>> obs, info = wrapped_env.reset()
    """

    def __init__(self, env):
        """
        初始化 Gym 包装器

        Args:
            env: Gymnasium 环境实例
        """
        super().__init__(env)
        self._spec = self._make_spec()

    def _make_spec(self) -> EnvSpec:
        """
        创建环境规格

        Returns:
            EnvSpec 对象
        """
        # 获取 Gym 环境的元数据
        gym_spec = getattr(self._env, 'spec', None)
        name = gym_spec.id if gym_spec else self._env.__class__.__name__

        # 最大步数
        max_steps = getattr(self._env, '_max_episode_steps', None)

        return EnvSpec(
            name=name,
            observation_space=GymSpace(self._env.observation_space),
            action_space=GymSpace(self._env.action_space),
            max_episode_steps=max_steps
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        重置环境

        Args:
            seed: 随机种子

        Returns:
            (初始观察, 信息字典)
        """
        # Gymnasium 的 reset 接口
        if seed is not None:
            obs, info = self._env.reset(seed=seed)
        else:
            obs, info = self._env.reset()
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步

        Args:
            action: 要执行的动作

        Returns:
            (观察, 奖励, terminated, truncated, 信息)
        """
        return self._env.step(action)

    @property
    def spec(self) -> EnvSpec:
        """获取环境规格"""
        return self._spec

    def __repr__(self) -> str:
        return f"GymWrapper({self._spec.name})"


def make_gym_env(env_id: str, render_mode: Optional[str] = None) -> GymWrapper:
    """
    便捷函数：创建并包装 Gym 环境

    Args:
        env_id: 环境 ID，如 "CartPole-v1"
        render_mode: 渲染模式，如 "human", "rgb_array"

    Returns:
        包装后的环境

    Examples:
        >>> env = make_gym_env("CartPole-v1")
        >>> obs, info = env.reset()
        >>> action = env.spec.action_space.sample()
        >>> obs, reward, done, truncated, info = env.step(action)
    """
    import gymnasium as gym

    raw_env = gym.make(env_id, render_mode=render_mode)
    return GymWrapper(raw_env)
