"""
=============================================
REINFORCE 策略梯度示例
=============================================

REINFORCE 是最基础的策略梯度算法。

与 DQN 的区别：
- DQN：学习每个动作的价值 Q(s,a)，然后选最大的
- REINFORCE：直接学习策略 π(a|s)，输出动作概率分布

REINFORCE 特点：
- On-policy：只能用当前策略收集的数据学习
- 每个 episode 结束后更新一次
- 不需要经验池
- 高方差，可能需要更多训练时间
"""

import torch
from prometheus.envs.gym_wrapper import make_gym_env
from prometheus.agents.policy_gradient.reinforce import REINFORCEAgent
from prometheus.trainers.policy_gradient.reinforce import REINFORCETrainer, TrainerConfig
from prometheus.core import Config


def main():
    # === 创建环境 ===
    env = make_gym_env("CartPole-v1")

    # === 配置 ===
    agent_config = Config(
        LEARNING_RATE=1e-3,    # 策略梯度通常用较小学习率
        GAMMA=0.99,
    )

    trainer_config = TrainerConfig(
        max_episodes=1000,
        max_steps_per_episode=500,
        eval_interval=100,
        eval_episodes=10,
        save_interval=500,
        save_dir="checkpoints/reinforce",
        log_interval=10,
    )

    # === 创建智能体 ===
    state_dim = env.spec.observation_space.shape[0]
    action_dim = env.spec.action_space._gym_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=agent_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # === 创建训练器 ===
    trainer = REINFORCETrainer(config=trainer_config)

    # === 开始训练 ===
    results = trainer.train(env, agent)

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
