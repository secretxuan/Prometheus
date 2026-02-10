"""
=============================================
A2C 训练器
=============================================
"""

import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

from prometheus.trainers.base import BaseTrainer, TrainerConfig, Callback
from prometheus.agents.policy_gradient.a2c import A2CAgent
from prometheus.envs.base import EnvWrapper


class ProgressCallback(Callback):
    """
    进度打印回调
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
            policy_loss = metrics.get("policy_loss", 0)
            value_loss = metrics.get("value_loss", 0)
            entropy = metrics.get("entropy", 0)

            print(f"Episode {episode:4d} | "
                  f"得分: {metrics.get('score', 0):3.0f} | "
                  f"平均: {avg_score:5.1f} | "
                  f"P_Loss: {policy_loss:.3f} | "
                  f"V_Loss: {value_loss:.3f} | "
                  f"Ent: {entropy:.3f}")


class A2CTrainer(BaseTrainer):
    """
    A2C 训练器

    支持两种更新方式：
    1. Episode 结束后更新（默认）
    2. N-step 更新（每 n 步更新一次）
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
        agent: A2CAgent,
        n_step_update: bool = False
    ) -> Dict[str, Any]:
        """
        训练智能体

        Args:
            env: 环境
            agent: A2C 智能体
            n_step_update: 是否使用 n-step 更新

        Returns:
            训练结果
        """
        # 获取环境信息
        state_dim = env.spec.observation_space.shape[0]
        action_dim = env.spec.action_space._gym_space.n

        print("=" * 60)
        print("🏛️  Prometheus - A2C 训练")
        print("=" * 60)
        print(f"环境: {env.spec.name}")
        print(f"状态维度: {state_dim}")
        print(f"动作数量: {action_dim}")
        print(f"最大 Episode: {self.config.max_episodes}")
        print(f"更新方式: {'N-step' if n_step_update else 'Episode-end'}")
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
            metrics = self._run_episode(env, agent, n_step_update)
            metrics["duration"] = time.time() - episode_start

            # === 学习（episode 结束后）===
            if not n_step_update:
                learn_metrics = agent.learn()
                metrics.update(learn_metrics)

            for callback in self.config.callbacks:
                callback.on_episode_end(self, episode, metrics)

            # 定期评估
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(env, agent, render=False)
                print(f"  → 评估平均得分: {eval_metrics['mean_score']:.1f}")

            # 定期保存
            if episode % self.config.save_interval == 0:
                save_path = Path(self.config.save_dir) / f"a2c_ep{episode}.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(save_path))

        for callback in self.config.callbacks:
            callback.on_train_end(self)

        print()
        print("=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)

        return {"episodes": episode, "final_score": metrics.get("score", 0)}

    def _run_episode(self, env: EnvWrapper, agent: A2CAgent, n_step_update: bool) -> Dict:
        """运行一个 episode"""
        agent.reset()
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

            # N-step 更新
            if n_step_update and agent.should_update():
                agent.learn()

            state = next_state
            score += reward
            step += 1

        return {"score": score, "steps": step}

    def evaluate(
        self,
        env: EnvWrapper,
        agent: A2CAgent,
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
            agent.reset()
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

    def save_checkpoint(self, path: str, agent: A2CAgent):
        """保存检查点"""
        agent.save(path)

    def load_checkpoint(self, path: str, agent: A2CAgent):
        """加载检查点"""
        agent.load(path)
