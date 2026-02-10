#!/usr/bin/env python3
"""
=============================================
Prometheus 示例 #03: 使用新框架训练
=============================================

这个示例展示了如何使用 Prometheus v0.1.0 的新框架架构。

新架构的优势：
-------------
1. 模块化设计：环境、策略、智能体、训练器各司其职
2. 清晰的接口：易于扩展和替换组件
3. 统一的 API：不同算法使用相同的接口

使用方法：
---------
./run.sh examples/03_new_framework.py
"""

from prometheus.envs import make_gym_env
from prometheus.agents import DQNAgent
from prometheus.trainers import DQNTrainer, TrainerConfig


def main():
    """
    主函数：使用新框架训练 DQN
    """
    # === 1. 创建环境 ===
    env = make_gym_env("CartPole-v1")

    # === 2. 创建智能体 ===
    # 注意：新框架自动从环境获取维度信息
    state_dim = env.spec.observation_space.shape[0]
    action_dim = env.spec.action_space._gym_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    # === 3. 创建训练器 ===
    config = TrainerConfig(
        max_episodes=5000,          # 训练 500 轮
        max_steps_per_episode=5000, # 每轮最多 500 步
        eval_interval=100,         # 每 100 轮评估一次
        log_interval=10,           # 每 10 轮打印一次
        save_interval=500,         # 每 500 轮保存一次
    )

    trainer = DQNTrainer(config=config)

    # === 4. 开始训练 ===
    result = trainer.train(
        env=env,
        agent=agent,
        target_update_interval=10  # 每 10 轮更新目标网络
    )

    # === 5. 评估最终效果 ===
    print("\n=== 最终评估 ===")
    final_metrics = trainer.evaluate(env, agent, n_episodes=10)
    print(f"平均得分: {final_metrics['mean_score']:.1f}")
    print(f"标准差: {final_metrics['std_score']:.1f}")
    print(f"最高分: {final_metrics['max_score']:.0f}")

    # === 6. 保存模型 ===
    agent.save("checkpoints/final_model.pth")
    print("\n模型已保存到: checkpoints/final_model.pth")

    env.close()

    return result


if __name__ == "__main__":
    main()
