#!/usr/bin/env python3
"""
=============================================
Prometheus 示例 #02: 观看训练后的智能体
=============================================

这个脚本会加载训练好的模型（或者快速训练一个），
然后用动画展示智能体如何平衡杆子。
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

# 导入之前定义的类和配置
import sys
sys.path.append('.')
from examples.train import QNetwork, DQNAgent, Config


def watch_human_mode(agent, env, episodes=5):
    """
    以人类可视化的方式观看智能体表现

    render_mode="human" 会打开一个窗口显示动画
    """
    print("\n=== 观看模式 ===")
    print("如果看不到窗口，可能是环境配置问题")

    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            # 选择动作（不探索，只利用）
            action = agent.select_action(state, training=False)

            # 执行动作
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward

        print(f"Episode {episode + 1}: 得分 = {score}")

    env.close()


def quick_train_and_watch():
    """快速训练一个模型然后观看"""

    print("=== 快速训练一个模型 (50 episodes) ===")

    # 创建环境（不渲染，训练更快）
    env = gym.make(Config.ENV_NAME)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建智能体
    agent = DQNAgent(state_dim, action_dim, Config)

    # 快速训练
    for episode in range(50):
        state, _ = env.reset()
        score = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= Config.BATCH_SIZE:
                agent.train()

            state = next_state
            score += reward

            if done:
                break

        if episode % 10 == 0:
            print(f"Episode {episode}: 得分 = {score}")

        if episode % 10 == 0:
            agent.update_target_network()

    env.close()

    print("\n=== 训练完成，现在观看效果 ===")

    # 创建带渲染的环境
    env = gym.make(Config.ENV_NAME, render_mode="human")

    # 观看
    watch_human_mode(agent, env, episodes=3)


if __name__ == "__main__":
    quick_train_and_watch()
