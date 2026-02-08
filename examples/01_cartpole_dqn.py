#!/usr/bin/env python3
"""
=============================================
Prometheus 示例 #01: DQN 解决 CartPole
=============================================

这是强化学习的 "Hello World"！

本文件实现了一个完整的 DQN (Deep Q-Network) 算法，
用于解决 CartPole（小车倒立摆）任务。

学习路线：
1. 先通读一遍代码，理解大致流程
2. 运行代码，看效果
3. 对照注释，理解每一步
4. 修改参数，观察变化

什么是 CartPole？
----------------
一个经典的控制任务：
- 有一辆小车，可以在水平轨道上左右移动
- 小车上方竖立一根杆子，杆子可以自由摆动
- 目标：通过移动小车，保持杆子不倒
- 如果杆子倾斜超过 15 度，或小车移出轨道，游戏结束

什么是 DQN？
-----------
DQN (Deep Q-Network) 是深度强化学习的奠基性算法，
由 DeepMind 在 2015 年发表在 Nature 上。

核心思想：
1. 用一个神经网络来预测 "在每种状态下，每个动作的价值"
2. 通过和环境的交互不断改进这个网络
3. 使用 "经验回放" 来提高样本利用效率

代码结构：
---------
- 核心类 (QNetwork, DQNAgent, Config) 在 prometheus.core 中
- 本文件专注于训练逻辑和可视化
"""

# ============================================================
# 导入必要的库
# ============================================================

import gymnasium as gym

# 导入 Prometheus 框架的核心类
from prometheus.core import DQNAgent, Config


# ============================================================
# 训练函数
# ============================================================

def train():
    """
    主训练循环

    训练流程：
    1. 创建环境和智能体
    2. 对每个 episode：
       a. 重置环境
       b. 对每个步骤：
          - 选择动作
          - 执行动作，获得反馈
          - 存储经验
          - 训练网络
    3. 定期更新目标网络
    """

    # === 创建环境 ===
    env = gym.make(Config.ENV_NAME, render_mode=None)  # 不渲染，训练更快

    # 获取环境信息
    state_dim = env.observation_space.shape[0]   # CartPole: 4
    action_dim = env.action_space.n              # CartPole: 2

    print(f"=== 环境信息 ===")
    print(f"环境名称: {Config.ENV_NAME}")
    print(f"状态维度: {state_dim}")
    print(f"动作数量: {action_dim}")
    print(f"状态含义: 小车位置、小车速度、杆子角度、杆子角速度")
    print(f"动作含义: 0=向左推, 1=向右推")
    print()

    # === 创建智能体 ===
    agent = DQNAgent(state_dim, action_dim, Config)

    # === 训练循环 ===
    scores = []  # 记录每个 episode 的得分
    avg_scores = []  # 记录平均得分

    print("=== 开始训练 ===")
    print(f"总 Episode 数: {Config.EPISODES}")
    print()

    for episode in range(1, Config.EPISODES + 1):
        # 重置环境
        state, _ = env.reset()  # state 是初始状态
        score = 0  # 本 episode 的总奖励
        done = False

        # === 一个 episode ===
        while not done:
            # 1. 选择动作
            action = agent.select_action(state, training=True)

            # 2. 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 存储经验
            agent.store_experience(state, action, reward, next_state, done)

            # 4. 训练（如果经验池足够）
            if len(agent.replay_buffer) >= Config.BATCH_SIZE:
                agent.train()

            # 5. 更新状态
            state = next_state
            score += reward

        # === Episode 结束后的处理 ===
        scores.append(score)

        # 计算最近 10 个 episode 的平均得分
        if len(scores) >= 10:
            avg_score = sum(scores[-10:]) / 10
            avg_scores.append(avg_score)
        else:
            avg_scores.append(sum(scores) / len(scores))

        # 每 10 个 episode 更新一次目标网络
        if episode % 10 == 0:
            agent.update_target_network()

        # === 打印进度 ===
        if episode % 10 == 0:
            print(f"Episode {int(episode):3d} | "
                  f"得分: {int(score):3d} | "
                  f"平均得分: {avg_scores[-1]:5.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"经验池: {len(agent.replay_buffer)}")

        # === 检查是否成功 ===
        # CartPole-v1 的成功标准是平均得分 >= 475
        if len(avg_scores) >= 10 and avg_scores[-1] >= 475:
            print(f"\n🎉 恭喜！在第 {episode} 个 episode 达到成功标准！")
            print(f"   平均得分: {avg_scores[-1]:.1f} >= 475")
            break

    env.close()

    return scores, avg_scores


# ============================================================
# 可视化结果
# ============================================================

def plot_results(scores, avg_scores):
    """
    绘制训练曲线

    Args:
        scores: 每个 episode 的得分
        avg_scores: 每个 episode 的平均得分
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        from matplotlib import font_manager

        # === 配置中文字体 ===
        # macOS 系统自带的中文字体
        mac_fonts = ['PingFang SC', 'Arial Unicode MS', 'STHeiti', 'Heiti TC']
        # Linux 常见中文字体
        linux_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
        # Windows 中文字体
        windows_fonts = ['Microsoft YaHei', 'SimHei']

        # 按优先级尝试设置字体
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]

        for font_list in [mac_fonts, linux_fonts, windows_fonts]:
            for font_name in font_list:
                if font_name in available_fonts:
                    matplotlib.rcParams['font.sans-serif'] = [font_name]
                    break
            else:
                continue
            break

        # 解决负号显示问题
        matplotlib.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 5))

        # 子图1: 得分曲线
        plt.subplot(1, 2, 1)
        plt.plot(scores, alpha=0.6, label='单次得分')
        plt.plot(avg_scores, linewidth=2, label='平均得分（10ep）')
        plt.axhline(y=475, color='r', linestyle='--', label='成功线 (475)')
        plt.xlabel('Episode')
        plt.ylabel('得分')
        plt.title('训练进度')
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图2: 最终得分分布
        plt.subplot(1, 2, 2)
        if len(scores) >= 50:
            recent_scores = scores[-50:]
        else:
            recent_scores = scores
        plt.hist(recent_scores, bins=20, edgecolor='black')
        plt.xlabel('得分')
        plt.ylabel('次数')
        plt.title('得分分布（最近）')
        plt.axvline(x=475, color='r', linestyle='--', label='成功线')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('examples/training_results.png', dpi=100)
        print("\n📊 训练曲线已保存到: examples/training_results.png")
    except Exception as e:
        print(f"\n⚠️  绘图失败: {e}")
        print("   请确保安装了 matplotlib: pip install matplotlib")


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数 - 程序入口
    """
    print("=" * 60)
    print("🏛️  Prometheus - DQN 训练示例")
    print("=" * 60)
    print()

    # 训练
    scores, avg_scores = train()

    print()
    print("=" * 60)
    print("📊 训练完成！")
    print(f"   最终平均得分: {avg_scores[-1]:.1f}")
    print(f"   最高得分: {max(scores)}")
    print("=" * 60)

    # 绘图
    plot_results(scores, avg_scores)


if __name__ == "__main__":
    main()
