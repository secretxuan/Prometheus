"""
=============================================
PPO（Proximal Policy Optimization）示例
=============================================

PPO 是目前最流行、最实用的强化学习算法之一。

核心特点：
1. 使用 Clipped Surrogate Objective 防止策略更新过大
2. 一批数据可以多次使用（高效）
3. 简单、稳定、高效

PPO vs 其他算法：
- REINFORCE：高方差，需要完整 episode
- A2C：方差更低，但可能不稳定
- PPO：稳定且高效，是很多项目的首选

适用场景：
- 连续动作空间和离散动作空间都适用
- 需要稳定训练的场景
- 需要样本效率的场景
"""

import torch
from prometheus.envs.gym_wrapper import make_gym_env
from prometheus.agents.policy_gradient.ppo import PPOAgent
from prometheus.trainers.policy_gradient.ppo import PPOTrainer, TrainerConfig
from prometheus.core import Config


def main():
    # === 创建环境 ===
    env = make_gym_env("CartPole-v1")

    # === 配置 ===
    # 设置全局 Config（静态类，直接修改属性）
    Config.LEARNING_RATE = 3e-4     # PPO 通常使用较小学习率
    Config.GAMMA = 0.99

    trainer_config = TrainerConfig(
        max_episodes=3000,
        max_steps_per_episode=500,
        eval_interval=50,
        eval_episodes=10,
        save_interval=150,
        save_dir="checkpoints/ppo",
        log_interval=10,
    )

    # === 创建智能体 ===
    state_dim = env.spec.observation_space.shape[0]
    action_dim = env.spec.action_space._gym_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # === 创建训练器 ===
    trainer = PPOTrainer(config=trainer_config)

    # === 开始训练 ===
    # PPO 参数
    n_epochs = 4          # 每次收集数据后更新多少轮
    batch_size = 64       # 批量大小

    trainer.train(env, agent, n_epochs=n_epochs, batch_size=batch_size)

    # === 最终评估 ===
    print("\n" + "=" * 60)
    print("📊 最终评估")
    print("=" * 60)
    final_metrics = trainer.evaluate(env, agent, n_episodes=20)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.2f}")

    env.close()


if __name__ == "__main__":
    main()
