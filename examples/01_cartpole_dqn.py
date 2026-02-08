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
"""

# ============================================================
# 第一部分：导入必要的库
# ============================================================

import torch              # PyTorch - 深度学习框架
import torch.nn as nn     # nn 包含了构建神经网络的模块
import torch.optim as optim  # optim 包含了各种优化器
import numpy as np        # NumPy - 数值计算库
import collections        # 提供了 deque 等有用的数据结构
import random            # 随机数生成
from collections import deque

# Gymnasium - 强化学习环境的标准接口
# 提供了各种 RL 环境，从简单的 CartPole 到复杂的 Atari 游戏
import gymnasium as gym


# ============================================================
# 第二部分：超参数配置
# ============================================================
"""
超参数是什么？
--------------
超参数是算法中需要人工设定的参数，不是通过学习得到的。
不同的超参数会显著影响算法性能。

这里每个参数都有详细解释，建议尝试修改它们看看效果！
"""

class Config:
    """超参数配置"""

    # === 环境相关 ===
    ENV_NAME = "CartPole-v1"     # 环境名称
    # v1 版本最大步数是 500，v0 版本是 200

    # === 训练相关 ===
    EPISODES = 500              # 总共训练多少局（一轮游戏叫一个 episode）
    MAX_STEPS = 500             # 每局最多走多少步

    # === DQN 核心参数 ===
    GAMMA = 0.99                # 折扣因子 (γ)
    """
    折扣因子解释：
    - 范围：0 到 1
    - 含义：未来的奖励在当前看来值多少
    - γ = 0.99：明天的 1 块钱 ≈ 今天的 0.99 块
    - γ 越大，越看重长期奖励；γ 越小，越看重短期奖励
    """

    LEARNING_RATE = 0.001       # 学习率 (α)
    """
    学习率解释：
    - 范围：0 到 1，通常很小如 0.001、0.0001
    - 含义：每次更新参数时，迈多大步子
    - 太大：可能不稳定，无法收敛
    - 太小：学习太慢
    """

    BATCH_SIZE = 64             # 批次大小
    """
    批次大小解释：
    - 每次训练神经网络时，用多少个样本
    - 太小：训练不稳定，梯度噪声大
    - 太大：内存占用大，训练慢
    """

    # === 经验回放相关 ===
    MEMORY_SIZE = 10000         # 经验池容量
    """
    经验池解释：
    - 存储过去的 "经验" (状态、动作、奖励、下一状态)
    - 训练时随机抽取，打破数据之间的相关性
    - 类比：人类的经验是随时间积累的，回忆时也是随机想起
    """

    # === 探索相关 ===
    EPSILON_START = 1.0         # 初始探索率
    EPSILON_END = 0.01          # 最终探索率
    EPSILON_DECAY = 0.995       # 探索率衰减
    """
    ε-贪婪策略解释：
    - ε (epsilon)：探索的概率
    - ε = 1.0：100% 随机探索（刚开始什么都不懂，多尝试）
    - ε = 0.01：1% 随机探索（基本学会了，偶尔探索）
    - 每次训练后：ε = ε × 0.995，逐渐减小
    """


# ============================================================
# 第三部分：经验回放缓冲区
# ============================================================

class ReplayBuffer:
    """
    经验回放缓冲区

    什么是经验回放？
    ---------------
    在传统的 Q-learning 中，每个样本只用一次就丢弃了。
    经验回放的思路是：把所有经验存起来，训练时随机抽取。
    这样的好处：
    1. 数据利用率高
    2. 打破了时间相关性（相邻的状态很相似）
    3. 更符合 i.i.d. 假设（独立同分布）

    数据结构解释：
    ------------
    每条 "经验" 是一个五元组：
        (state, action, reward, next_state, done)

    - state: 当前状态（比如：小车位置、速度、杆子角度、角速度）
    - action: 采取的动作（比如：向左推 or 向右推）
    - reward: 获得的奖励（比如：+1 每存活一步）
    - next_state: 采取动作后的新状态
    - done: 是否结束（游戏是否结束）
    """

    def __init__(self, capacity: int):
        """
        初始化经验池

        Args:
            capacity: 能存多少条经验
        """
        self.buffer = deque(maxlen=capacity)  # deque 是双端队列，超出容量自动删除旧数据

    def push(self, state, action, reward, next_state, done):
        """
        存入一条经验

        Args:
            state: 当前状态 (numpy array)
            action: 采取的动作 (int)
            reward: 获得的奖励 (float)
            next_state: 下一状态 (numpy array)
            done: 是否结束 (bool)
        """
        # 将经验打包成 tuple 存入
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        随机采样一批经验

        Args:
            batch_size: 要采样的数量

        Returns:
            五个 batch 的 tuple
        """
        # 从 buffer 中随机抽取 batch_size 条经验
        batch = random.sample(self.buffer, batch_size)

        # 将 batch 拆分成五个独立的列表
        # zip(*batch) 的作用类似矩阵转置
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),       # shape: [batch_size, state_dim]
            np.array(actions),      # shape: [batch_size]
            np.array(rewards),      # shape: [batch_size]
            np.array(next_states),  # shape: [batch_size, state_dim]
            np.array(dones)         # shape: [batch_size]
        )

    def __len__(self):
        """返回当前存储的经验数量"""
        return len(self.buffer)


# ============================================================
# 第四部分：Q 网络（神经网络）
# ============================================================

class QNetwork(nn.Module):
    """
    Q 网络 - DQN 的核心

    什么是 Q 网络？
    --------------
    Q 网络是一个函数 approximator（近似器），用来学习 Q 函数。

    Q 函数 Q(s, a) 的含义：
    - 在状态 s 下，采取动作 a，之后按照最优策略行动，能获得的期望回报

    直观理解：
    - Q(小车向左倾斜，向左推) = 很高的分数
    - Q(小车向左倾斜，向右推) = 很低的分数

    为什么用神经网络？
    ----------------
    - 状态空间很大（甚至连续），无法用表格存储所有 Q 值
    - 神经网络是"万能函数拟合器"，可以近似任意函数
    - 状态相似时，Q 值也应该相似（泛化能力）
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化 Q 网络

        Args:
            state_dim: 状态的维度（CartPole 是 4）
            action_dim: 动作的数量（CartPole 是 2：左/右）
            hidden_dim: 隐藏层维度
        """
        super(QNetwork, self).__init__()  # 调用父类初始化

        # 构建一个简单的前馈神经网络
        # 输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层

        self.network = nn.Sequential(
            # 第一层：输入层 -> 隐藏层
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),  # 激活函数，引入非线性

            # 第二层：隐藏层 -> 隐藏层
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            # 输出层：隐藏层 -> 动作数量
            nn.Linear(hidden_dim, action_dim)
            # 注意：这里没有激活函数，直接输出 Q 值（可以是任意实数）
        )

    def forward(self, state):
        """
        前向传播

        Args:
            state: 状态张量，shape [batch_size, state_dim]

        Returns:
            Q 值，shape [batch_size, action_dim]
            每个状态对应每个动作的 Q 值
        """
        return self.network(state)


# ============================================================
# 第五部分：DQN 智能体
# ============================================================

class DQNAgent:
    """
    DQN 智能体

    什么是智能体 (Agent)？
    --------------------
    智能体是强化学习中的"决策者"，它：
    1. 观察环境状态
    2. 做出决策（选择动作）
    3. 从环境中获得奖励
    4. 根据奖励更新自己的策略

    这里的 DQN 智能体包含：
    - Q 网络：预测每个动作的价值
    - 经验池：存储和回放经验
    - 优化器：更新网络参数
    """

    def __init__(self, state_dim: int, action_dim: int, config: Config):
        """
        初始化 DQN 智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            config: 超参数配置
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # 创建 Q 网络
        self.q_network = QNetwork(state_dim, action_dim)

        # 创建目标网络（用于稳定训练）
        # 目标网络的解释见 train() 方法
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # 初始时，目标网络和主网络完全一样

        # 创建优化器 - Adam 是最常用的优化器之一
        # 它自适应地调整每个参数的学习率
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE
        )

        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config.MEMORY_SIZE)

        # 探索率 ε
        self.epsilon = config.EPSILON_START

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-贪婪策略）

        ε-贪婪策略：
        1. 以 ε 的概率随机选择动作（探索）
        2. 以 1-ε 的概率选择 Q 值最大的动作（利用）

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            选择的动作（整数索引）
        """
        # === 探索阶段 ===
        if training and random.random() < self.epsilon:
            # 随机选择一个动作
            return random.randrange(self.action_dim)

        # === 利用阶段 ===
        # 将 numpy array 转换为 PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0)  # unsqueze(0) 增加 batch 维度
        # shape: [state_dim] -> [1, state_dim]

        # 计算每个动作的 Q 值
        with torch.no_grad():  # 不需要计算梯度，只是为了预测
            q_values = self.q_network(state)
            # q_values.shape: [1, action_dim]

        # 选择 Q 值最大的动作
        # argmax(1) 表示在 action_dim 维度上取最大值的索引
        action = q_values.argmax(1).item()
        # .item() 将 tensor 转换为 Python 标量

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """存储一条经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        """
        训练 Q 网络（一次更新步骤）

        DQN 的核心训练逻辑：
        1. 从经验池中随机采样一批经验
        2. 计算目标 Q 值（使用目标网络）
        3. 计算预测 Q 值（使用主网络）
        4. 计算损失，反向传播更新主网络
        5. 定期同步目标网络

        DQN 损失函数：
        L = (Q(s,a) - (r + γ × max Q'(s', a')))²
           ↑              ↑
        预测值         目标值

        Q(s,a): 主网络预测的 Q 值
        r: 实际获得的奖励
        γ: 折扣因子
        Q'(s', a'): 目标网络预测的下一状态的 Q 值
        """
        # 如果经验池不够，不训练
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return

        # === 1. 采样经验 ===
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.BATCH_SIZE
        )

        # 转换为 PyTorch tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)  # 动作是整数，用 LongTensor
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # === 2. 计算目标 Q 值 ===
        # Q_target = reward + γ × max Q_target(next_state) × (1 - done)

        # 计算下一状态每个动作的 Q 值（用目标网络）
        with torch.no_grad():  # 目标网络不需要梯度
            next_q_values = self.target_network(next_states)
            # shape: [batch_size, action_dim]

            # 取最大值
            next_q_max = next_q_values.max(1)[0]
            # max(1)[0] 表示在第 1 维度上取最大值，并返回值而非索引
            # shape: [batch_size]

            # 计算目标 Q 值
            # done=1 时表示游戏结束，没有未来奖励
            # done=0 时表示继续，有未来奖励
            target_q_values = rewards + self.config.GAMMA * next_q_max * (1 - dones)
            # shape: [batch_size]

        # === 3. 计算当前 Q 值 ===
        # 我们需要的是 "在状态 s 下，选择动作 a 的 Q 值"
        # gather 操作用于从多个 Q 值中取出特定动作的 Q 值

        # 先计算所有动作的 Q 值
        current_q_values = self.q_network(states)
        # shape: [batch_size, action_dim]

        # 取出实际采取的动作的 Q 值
        # actions.unsqueeze(1) 将 [batch_size] 变成 [batch_size, 1]
        # gather(1, actions.unsqueeze(1)) 沿 action_dim 维度收集
        action_indices = actions.unsqueeze(1)
        q_values = current_q_values.gather(1, action_indices).squeeze(1)
        # shape: [batch_size]

        # === 4. 计算损失并更新 ===
        # 使用 Huber Loss（Smooth L1 Loss），比 MSE 更鲁棒
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()             # 计算梯度
        self.optimizer.step()       # 更新参数

        # === 5. 更新探索率 ===
        self.epsilon = max(
            self.config.EPSILON_END,
            self.epsilon * self.config.EPSILON_DECAY
        )

        return loss.item()

    def update_target_network(self):
        """
        将主网络的参数复制到目标网络

        为什么要用目标网络？
        ------------------
        这是一个训练稳定性的技巧。

        问题：如果只有一个网络，目标值和预测值都来自同一个网络，
        训练会不稳定（"追逐移动的目标"）。

        解决：使用一个"冻结"的目标网络，每隔一段时间才更新。
        这样目标值在一段时间内是固定的，训练更稳定。

        类比：打靶时，靶子不动比靶子一直移动更容易瞄准。
        """
        self.target_network.load_state_dict(self.q_network.state_dict())


# ============================================================
# 第六部分：训练循环
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
    # render_mode="human" 可以看到动画，但会慢很多

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
        loss = None

        # === 一个 episode ===
        for step in range(Config.MAX_STEPS):
            # 1. 选择动作
            action = agent.select_action(state, training=True)

            # 2. 执行动作
            # env.step() 返回:
            # - next_state: 新状态
            # - reward: 奖励
            # - terminated: 是否成功结束（如杆子倒了）
            # - truncated: 是否被截断（如超过最大步数）
            # - info: 额外信息
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 存储经验
            agent.store_experience(state, action, reward, next_state, done)

            # 4. 训练（如果经验池足够）
            if len(agent.replay_buffer) >= Config.BATCH_SIZE:
                loss = agent.train()

            # 5. 更新状态
            state = next_state
            score += reward

            # 6. 如果结束，跳出循环
            if done:
                break

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
# 第七部分：可视化结果
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
# 第八部分：主程序
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
