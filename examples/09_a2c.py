"""
=============================================
A2C（Actor-Critic）示例
=============================================

A2C 是 Actor-Critic 方法的一个高效实现。

Actor-Critic 的核心思想：
1. Actor（演员）：策略网络，选择动作
2. Critic（评论家）：价值网络，评估状态价值

相比 REINFORCE 的优势：
1. 方差更低（Critic 提供了基线）
2. 收敛更快
3. 可以在线更新（不需要等 episode 结束）

A2C 的特点：
- 使用 Advantage 函数：A(s,a) = Q(s,a) - V(s)
- 同步更新（相比 A3C 的异步更新）
"""

import torch
from prometheus.envs.gym_wrapper import make_gym_env
from prometheus.agents.policy_gradient.a2c import A2CAgent
from prometheus.trainers.policy_gradient.a2c import A2CTrainer, TrainerConfig
from prometheus.core import Config


def main():
    # === 创建环境 ===
    env = make_gym_env("CartPole-v1")

    # === 配置 ===
    agent_config = Config(
        LEARNING_RATE=1e-3,
        GAMMA=0.99,
    )

    trainer_config = TrainerConfig(
        max_episodes=500,
        max_steps_per_episode=500,
        eval_interval=50,
        eval_episodes=10,
        save_interval=200,
        save_dir="checkpoints/a2c",
        log_interval=10,
    )

    # === 创建智能体 ===
    state_dim = env.spec.observation_space.shape[0]
    action_dim = env.spec.action_space._gym_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=agent_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # === 创建训练器 ===
    trainer = A2CTrainer(config=trainer_config)

    # === 开始训练 ===
    results = trainer.train(env, agent, n_step_update=False)

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
