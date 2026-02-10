"""
=============================================
Rainbow DQN 示例
=============================================

Rainbow 整合了 DQN 的多种改进：
- Dueling DQN 网络结构
- Double DQN 的目标 Q 计算
- 优先级经验回放（PER）

论文: Rainbow: Combining Improvements in Deep Reinforcement Learning (2017)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from prometheus.envs import make_gym_env
from prometheus.agents.dqn import (
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    PERAgent,
    RainbowAgent
)
from prometheus.trainers import DQNTrainer, TrainerConfig


def train_rainbow():
    """训练 Rainbow 智能体"""
    print("=" * 60)
    print("🌈 训练 Rainbow DQN")
    print("=" * 60)
    print("\nRainbow 整合的改进:")
    print("  ✓ Dueling DQN: Q(s,a) = V(s) + A(s,a)")
    print("  ✓ Double DQN: 主网络选动作，目标网络评估")
    print("  ✓ PER: 按优先级采样经验")

    env = make_gym_env("CartPole-v1")
    agent = RainbowAgent(state_dim=4, action_dim=2)
    config = TrainerConfig(max_episodes=300, eval_interval=50, log_interval=20)
    trainer = DQNTrainer(config)

    result = trainer.train(env, agent)
    print(f"\nRainbow 最终得分: {result['final_score']:.1f}")
    return agent, result


def compare_all():
    """对比所有 DQN 变体"""
    print("\n" + "=" * 60)
    print("🔬 DQN 系列算法对比")
    print("=" * 60)

    config = TrainerConfig(
        max_episodes=200,
        eval_interval=100,
        log_interval=50,
        save_interval=1000
    )

    results = {}

    algorithms = [
        ("标准 DQN", lambda: DQNAgent(state_dim=4, action_dim=2)),
        ("Double DQN", lambda: DoubleDQNAgent(state_dim=4, action_dim=2)),
        ("Dueling DQN", lambda: DuelingDQNAgent(state_dim=4, action_dim=2)),
        ("PER", lambda: PERAgent(state_dim=4, action_dim=2)),
        ("Rainbow", lambda: RainbowAgent(state_dim=4, action_dim=2)),
    ]

    for name, agent_fn in algorithms:
        print(f"\n--- {name} ---")
        env = make_gym_env("CartPole-v1")
        agent = agent_fn()
        trainer = DQNTrainer(config)
        trainer.train(env, agent)
        eval_result = trainer.evaluate(env, agent, n_episodes=20)
        results[name] = eval_result['mean_score']
        print(f"\n{name} 评估得分: {eval_result['mean_score']:.1f} ± {eval_result['std_score']:.1f}")

    # 总结
    print("\n" + "=" * 60)
    print("📈 对比结果（按得分排序）")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_results, 1):
        bar = "█" * int(score / 20)
        print(f"{i}. {name:15s} {score:5.1f} {bar}")

    print(f"\n🏆 最佳算法: {sorted_results[0][0]}")


def analyze_rainbow_components():
    """分析 Rainbow 的各个组件"""
    print("=" * 60)
    print("🔍 Rainbow 组件分析")
    print("=" * 60)

    agent = RainbowAgent(state_dim=4, action_dim=2)

    print("\n1. Dueling 网络结构:")
    print("   Q(s,a) = V(s) + A(s,a) - mean(A)")

    state = np.array([0, 0, 0, 0], dtype=np.float32)
    value, advantage = agent.policy.get_value_and_advantage(state)
    q_values = agent.policy.get_q_values(state)

    print(f"   状态价值 V(s): {value:.4f}")
    print(f"   动作优势 A(s,a): {advantage}")
    print(f"   Q 值: {q_values}")

    print("\n2. Double DQN 目标计算:")
    print("   next_action = policy_network(next_state).argmax()")
    print("   target_q = target_network(next_state)[next_action]")

    print("\n3. 优先级经验回放 (PER):")
    print(f"   Alpha (优先级指数): {agent.replay_buffer.alpha}")
    print(f"   Beta Start (重要性采样): {agent.replay_buffer.beta_start}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rainbow DQN 示例")
    parser.add_argument("--mode", choices=["analyze", "train", "compare"],
                        default="compare", help="运行模式")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_rainbow_components()
    elif args.mode == "train":
        train_rainbow()
    else:
        compare_all()
