"""
=============================================
DQN 训练器
=============================================

专门训练 DQN 智能体的训练器。
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from prometheus.trainers.base import BaseTrainer, TrainerConfig, Callback
from prometheus.agents.dqn import DQNAgent
from prometheus.envs.base import EnvWrapper


class ProgressCallback(Callback):
    """
    进度打印回调

    每隔一定间隔打印训练进度
    """

    def __init__(self, interval: int = 10):
        self.interval = interval
        self.scores = []
        self.episode_times = []

    def on_episode_end(self, trainer, episode: int, metrics: Dict):
        self.scores.append(metrics.get("score", 0))
        self.episode_times.append(metrics.get("duration", 0))

        if episode % self.interval == 0:
            avg_score = np.mean(self.scores[-self.interval:])
            eps = metrics.get("epsilon", 0)
            buffer_size = metrics.get("buffer_size", 0)

            print(f"Episode {episode:4d} | "
                  f"得分: {metrics.get('score', 0):3.0f} | "
                  f"平均: {avg_score:5.1f} | "
                  f"ε: {eps:.3f} | "
                  f"经验池: {buffer_size}")


class DQNTrainer(BaseTrainer):
    """
    DQN 训练器

    通俗解释：
    -----------
    专门负责训练 DQN 智能体的"教练"

    使用方法：
    ---------
    >>> from prometheus.envs import make_gym_env
    >>> from prometheus.agents import DQNAgent
    >>> from prometheus.trainers import DQNTrainer
    >>>
    >>> env = make_gym_env("CartPole-v1")
    >>> agent = DQNAgent(state_dim=4, action_dim=2)
    >>> trainer = DQNTrainer()
    >>> result = trainer.train(env, agent)
    """

    def __init__(self, config: TrainerConfig = None):
        super().__init__(config)
        self._setup_callbacks()

    def _setup_callbacks(self):
        """设置默认回调"""
        if not self.config.callbacks:
            self.config.callbacks = [ProgressCallback(interval=self.config.log_interval)]

    def train(
        self,
        env: EnvWrapper,
        agent: DQNAgent,
        target_update_interval: int = 10
    ) -> Dict[str, Any]:
        """
        训练智能体

        Args:
            env: 环境
            agent: DQN 智能体
            target_update_interval: 每隔多少 episode 更新目标网络

        Returns:
            训练结果
        """
        # 获取环境信息
        state_dim = env.spec.observation_space.shape[0]
        action_dim = env.spec.action_space.shape if env.spec.action_space.shape else (env.spec.action_space,)
        if len(action_dim) == 0:
            action_dim = env.spec.action_space.shape  # 这可能需要调整
        # 简化处理：对于离散动作空间
        if hasattr(env.spec.action_space._gym_space, 'n'):
            action_dim = env.spec.action_space._gym_space.n

        print("=" * 60)
        print("🏛️  Prometheus - DQN 训练")
        print("=" * 60)
        print(f"环境: {env.spec.name}")
        print(f"状态维度: {state_dim}")
        print(f"动作数量: {action_dim}")
        print(f"最大 Episode: {self.config.max_episodes}")
        print("=" * 60)
        print()

        # 训练循环
        for callback in self.config.callbacks:
            callback.on_train_start(self)

        for episode in range(1, self.config.max_episodes + 1):
            episode_start = time.time()

            for callback in self.config.callbacks:
                callback.on_episode_start(self, episode)

            # 运行一个 episode
            metrics = self._run_episode(env, agent)
            metrics["duration"] = time.time() - episode_start
            metrics["epsilon"] = agent._epsilon
            metrics["buffer_size"] = len(agent.replay_buffer)

            for callback in self.config.callbacks:
                callback.on_episode_end(self, episode, metrics)

            # 定期更新目标网络
            if episode % target_update_interval == 0:
                agent.update_target_network()

            # 定期评估
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(env, agent, render=False)
                print(f"  → 评估平均得分: {eval_metrics['mean_score']:.1f}")

            # 定期保存
            if episode % self.config.save_interval == 0:
                save_path = Path(self.config.save_dir) / f"checkpoint_ep{episode}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(save_path))

        for callback in self.config.callbacks:
            callback.on_train_end(self)

        print()
        print("=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)

        return {"episodes": episode, "final_score": metrics.get("score", 0)}

    def _run_episode(self, env: EnvWrapper, agent: DQNAgent) -> Dict:
        """运行一个 episode"""
        state, _ = env.reset()
        score = 0
        done = False
        step = 0

        while not done and step < self.config.max_steps_per_episode:
            # 选择动作
            action = agent.act(state, training=True)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 学习
            if len(agent.replay_buffer) >= agent.config.BATCH_SIZE:
                agent.learn()

            state = next_state
            score += reward
            step += 1

        return {"score": score, "steps": step}

    def evaluate(
        self,
        env: EnvWrapper,
        agent: DQNAgent,
        n_episodes: int = None,
        render: bool = False
    ) -> Dict[str, float]:
        """
        评估智能体

        Args:
            env: 环境
            agent: 智能体
            n_episodes: 评估轮数
            render: 是否渲染

        Returns:
            评估结果
        """
        if n_episodes is None:
            n_episodes = self.config.eval_episodes

        scores = []
        agent.set_mode(training=False)

        for _ in range(n_episodes):
            state, _ = env.reset()
            score = 0
            done = False

            while not done:
                action = agent.act(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                score += reward

            scores.append(score)

        agent.set_mode(training=True)

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores)
        }

    def save_checkpoint(self, path: str, agent: DQNAgent):
        """保存检查点"""
        agent.save(path)

    def load_checkpoint(self, path: str, agent: DQNAgent):
        """加载检查点"""
        agent.load(path)
