#!/usr/bin/env python3
"""
=============================================
Prometheus 示例 #02: 观看训练后的智能体
=============================================

这个脚本会快速训练一个模型，然后用动画展示智能体如何平衡杆子。

运行方式:
    python examples/02_watch_agent.py
"""

import gymnasium as gym

# 导入框架核心类
from prometheus.core import DQNAgent, Config


def watch_human_mode(agent, env, episodes=5):
    """
    以人类可视化的方式观看智能体表现

    Args:
        agent: 训练好的智能体
        env: 环境（带 render_mode="human"）
        episodes: 要观看多少局
    """
    print("\n=== 观看模式 ===")
    print("提示: 如果看不到窗口，请确保系统支持 GUI")

    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            # 选择动作（training=False：不探索，只利用学到的策略）
            action = agent.select_action(state, training=False)

            # 执行动作
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward

        print(f"Episode {episode + 1}: 得分 = {int(score)}")

    env.close()


def quick_train_and_watch():
    """快速训练一个模型然后观看"""

    print("=== 快速训练一个模型 (500 episodes) ===")
    print("训练中...\n")

    # 创建环境（不渲染，训练更快）
    env = gym.make(Config.ENV_NAME)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建智能体
    agent = DQNAgent(state_dim, action_dim, Config)

    # 快速训练
    for episode in range(500):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= Config.BATCH_SIZE:
                agent.train()

            state = next_state
            score += reward

        # 每 10 个 episode 打印一次
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}: 得分 = {int(score):3d}, ε = {agent.epsilon:.3f}")

        # 每 10 个 episode 更新一次目标网络
        if (episode + 1) % 10 == 0:
            agent.update_target_network()

    env.close()

    print("\n=== 训练完成，现在观看效果 ===")
    print("正在打开可视化窗口...\n")

    # 创建带渲染的环境
    env = gym.make(Config.ENV_NAME, render_mode="human")

    # 观看 3 局
    watch_human_mode(agent, env, episodes=3)


if __name__ == "__main__":
    quick_train_and_watch()
