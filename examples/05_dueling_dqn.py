"""
=============================================
Dueling DQN 示例
=============================================

Dueling DQN 将 Q 值分解为状态价值和动作优势：
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from prometheus.envs import make_gym_env
from prometheus.agents.dqn import DQNAgent, DuelingDQNAgent
from prometheus.trainers import DQNTrainer, TrainerConfig


def analyze_dueling_network():
    """分析 Dueling 网络的输出"""
    print("=" * 60)
    print("🔬 Dueling DQN 网络结构分析")
    print("=" * 60)

    from prometheus.agents.dqn import DuelingDQNAgent
    from prometheus.core import Config

    agent = DuelingDQNAgent(state_dim=4, action_dim=2)

    # 模拟一个状态
    state = np.array([0, 0, 0, 0], dtype=np.float32)

    # 获取 Q 值
    q_values = agent.policy.get_q_values(state)
    print(f"\nQ 值: {q_values}")

    # 分别获取 V(s) 和 A(s,a)
    value, advantage = agent.policy.get_value_and_advantage(state)
    print(f"\n状态价值 V(s): {value:.4f}")
    print(f"动作优势 A(s,a): {advantage}")

    # 验证 Q = V + A - mean(A)
    q_computed = value + advantage - advantage.mean()
    print(f"\n验证 Q(s,a) = V(s) + A(s,a) - mean(A): {q_computed}")
    print(f"直接 Q(s,a):                     {q_values}")
    print(f"匹配: {np.allclose(q_values, q_computed)}")


def train_dueling_dqn():
    """训练 Dueling DQN"""
    print("\n" + "=" * 60)
    print("📊 训练 Dueling DQN")
    print("=" * 60)

    env = make_gym_env("CartPole-v1")
    agent = DuelingDQNAgent(state_dim=4, action_dim=2)
    config = TrainerConfig(max_episodes=300, eval_interval=50, log_interval=20)
    trainer = DQNTrainer(config)

    result = trainer.train(env, agent)
    print(f"\nDueling DQN 最终得分: {result['final_score']:.1f}")
    return agent, result


def compare_with_standard():
    """对比标准 DQN 和 Dueling DQN"""
    print("\n" + "=" * 60)
    print("🔬 标准 DQN vs Dueling DQN")
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

    # 训练 Dueling DQN
    print("\n--- Dueling DQN ---")
    env = make_gym_env("CartPole-v1")
    agent_dueling = DuelingDQNAgent(state_dim=4, action_dim=2)
    trainer_dueling = DQNTrainer(config)
    trainer_dueling.train(env, agent_dueling)
    eval_dueling = trainer_dueling.evaluate(env, agent_dueling, n_episodes=20)
    print(f"\nDueling DQN 评估得分: {eval_dueling['mean_score']:.1f} ± {eval_dueling['std_score']:.1f}")

    # 总结
    print("\n" + "=" * 60)
    print("📈 对比结果")
    print("=" * 60)
    print(f"标准 DQN:      {eval_standard['mean_score']:.1f}")
    print(f"Dueling DQN:   {eval_dueling['mean_score']:.1f}")
    improvement = eval_dueling['mean_score'] - eval_standard['mean_score']
    print(f"提升:          {improvement:+.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dueling DQN 示例")
    parser.add_argument("--mode", choices=["analyze", "train", "compare"],
                        default="compare", help="运行模式")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_dueling_network()
    elif args.mode == "train":
        train_dueling_dqn()
    else:
        compare_with_standard()
