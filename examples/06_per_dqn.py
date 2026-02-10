"""
=============================================
优先级经验回放（PER）示例
=============================================

PER 按优先级采样经验，重点学习"意外"的经验。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from prometheus.envs import make_gym_env
from prometheus.agents.dqn import DQNAgent, PERAgent
from prometheus.trainers import DQNTrainer, TrainerConfig
from prometheus.core import SumTree


def demonstrate_sumtree():
    """演示 SumTree 的使用"""
    print("=" * 60)
    print("🌳 SumTree 演示")
    print("=" * 60)

    tree = SumTree(capacity=8)

    # 添加一些数据
    for i in range(8):
        priority = i + 1  # 优先级 1, 2, 3, ..., 8
        tree.add(priority, f"data_{i}")

    print(f"\n优先级总和: {tree.total()}")  # 应该是 36
    print(f"存储数量: {tree.n_entries}")

    # 采样几次
    print("\n采样结果（模拟）:")
    for _ in range(5):
        s = np.random.uniform(0, tree.total())
        idx, priority, data = tree.get(s)
        print(f"  采样值 {s:.1f} -> 优先级 {priority:.0f}, 数据 {data}")

    # 更新优先级
    print("\n更新优先级（第一个数据从 1 改为 100）:")
    idx = 7  # 第一个叶子节点索引
    tree.update(idx, 100)
    print(f"新的总和: {tree.total()}")


def train_per():
    """训练 PER 智能体"""
    print("\n" + "=" * 60)
    print("📊 训练 PER 智能体")
    print("=" * 60)

    env = make_gym_env("CartPole-v1")
    agent = PERAgent(state_dim=4, action_dim=2, alpha=0.6, beta_start=0.4)
    config = TrainerConfig(max_episodes=300, eval_interval=50, log_interval=20)
    trainer = DQNTrainer(config)

    result = trainer.train(env, agent)
    print(f"\nPER 最终得分: {result['final_score']:.1f}")
    return agent, result


def compare_standard_vs_per():
    """对比标准 DQN 和 PER"""
    print("\n" + "=" * 60)
    print("🔬 标准 DQN vs PER")
    print("=" * 60)

    config = TrainerConfig(
        max_episodes=200,
        eval_interval=100,
        log_interval=50,
        save_interval=1000
    )

    # 训练标准 DQN
    print("\n--- 标准 DQN ---")
    env = make_gym_env("CartPole-v1")
    agent_standard = DQNAgent(state_dim=4, action_dim=2)
    trainer_standard = DQNTrainer(config)
    trainer_standard.train(env, agent_standard)
    eval_standard = trainer_standard.evaluate(env, agent_standard, n_episodes=20)
    print(f"\n标准 DQN 评估得分: {eval_standard['mean_score']:.1f} ± {eval_standard['std_score']:.1f}")

    # 训练 PER
    print("\n--- PER ---")
    env = make_gym_env("CartPole-v1")
    agent_per = PERAgent(state_dim=4, action_dim=2)
    trainer_per = DQNTrainer(config)
    trainer_per.train(env, agent_per)
    eval_per = trainer_per.evaluate(env, agent_per, n_episodes=20)
    print(f"\nPER 评估得分: {eval_per['mean_score']:.1f} ± {eval_per['std_score']:.1f}")

    # 总结
    print("\n" + "=" * 60)
    print("📈 对比结果")
    print("=" * 60)
    print(f"标准 DQN:      {eval_standard['mean_score']:.1f}")
    print(f"PER:           {eval_per['mean_score']:.1f}")
    improvement = eval_per['mean_score'] - eval_standard['mean_score']
    print(f"提升:          {improvement:+.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="优先级经验回放示例")
    parser.add_argument("--mode", choices=["sumtree", "train", "compare"],
                        default="compare", help="运行模式")
    args = parser.parse_args()

    if args.mode == "sumtree":
        demonstrate_sumtree()
    elif args.mode == "train":
        train_per()
    else:
        compare_standard_vs_per()
