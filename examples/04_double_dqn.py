"""
=============================================
Double DQN 示例
=============================================

Double DQN vs 标准 DQN 对比实验

Double DQN 的核心改进：
- 用主网络选择动作
- 用目标网络评估价值
- 解决 Q 值过高估计问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus.envs import make_gym_env
from prometheus.agents.dqn import DQNAgent, DoubleDQNAgent
from prometheus.trainers import DQNTrainer, TrainerConfig


def train_standard_dqn():
    """训练标准 DQN"""
    print("=" * 60)
    print("📊 训练标准 DQN")
    print("=" * 60)

    env = make_gym_env("CartPole-v1")
    agent = DQNAgent(state_dim=4, action_dim=2)
    config = TrainerConfig(max_episodes=300, eval_interval=50, log_interval=20)
    trainer = DQNTrainer(config)

    result = trainer.train(env, agent)
    print(f"\n标准 DQN 最终得分: {result['final_score']:.1f}")
    return agent, result


def train_double_dqn():
    """训练 Double DQN"""
    print("\n" + "=" * 60)
    print("📊 训练 Double DQN")
    print("=" * 60)

    env = make_gym_env("CartPole-v1")
    agent = DoubleDQNAgent(state_dim=4, action_dim=2)
    config = TrainerConfig(max_episodes=300, eval_interval=50, log_interval=20)
    trainer = DQNTrainer(config)

    result = trainer.train(env, agent)
    print(f"\nDouble DQN 最终得分: {result['final_score']:.1f}")
    return agent, result


def compare_agents():
    """对比两种算法"""
    print("\n" + "=" * 60)
    print("🔬 算法对比")
    print("=" * 60)

    # 训练标准 DQN
    print("\n--- 标准 DQN ---")
    env = make_gym_env("CartPole-v1")
    agent_standard = DQNAgent(state_dim=4, action_dim=2)
    config = TrainerConfig(
        max_episodes=200,
        eval_interval=100,
        log_interval=50,
        save_interval=1000  # 不保存
    )
    trainer_standard = DQNTrainer(config)
    trainer_standard.train(env, agent_standard)

    # 评估标准 DQN
    eval_standard = trainer_standard.evaluate(env, agent_standard, n_episodes=20)
    print(f"\n标准 DQN 评估得分: {eval_standard['mean_score']:.1f} ± {eval_standard['std_score']:.1f}")

    # 训练 Double DQN
    print("\n--- Double DQN ---")
    env = make_gym_env("CartPole-v1")
    agent_double = DoubleDQNAgent(state_dim=4, action_dim=2)
    trainer_double = DQNTrainer(config)
    trainer_double.train(env, agent_double)

    # 评估 Double DQN
    eval_double = trainer_double.evaluate(env, agent_double, n_episodes=20)
    print(f"\nDouble DQN 评估得分: {eval_double['mean_score']:.1f} ± {eval_double['std_score']:.1f}")

    # 总结
    print("\n" + "=" * 60)
    print("📈 对比结果")
    print("=" * 60)
    print(f"标准 DQN:    {eval_standard['mean_score']:.1f}")
    print(f"Double DQN:  {eval_double['mean_score']:.1f}")
    if eval_double['mean_score'] > eval_standard['mean_score']:
        print("✅ Double DQN 表现更好！")
    else:
        print("📊 两种算法表现相近")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Double DQN 示例")
    parser.add_argument("--mode", choices=["standard", "double", "compare"],
                        default="compare", help="运行模式")
    args = parser.parse_args()

    if args.mode == "standard":
        train_standard_dqn()
    elif args.mode == "double":
        train_double_dqn()
    else:
        compare_agents()
