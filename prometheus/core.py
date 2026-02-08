"""
=============================================
Prometheus 核心模块
=============================================

这里放框架的核心类定义，避免重复代码。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ============================================================
# 配置类 - 存放所有超参数
# ============================================================

class Config:
    """
    超参数配置类

    什么是超参数？
    -------------
    超参数就是我们需要手动设定的参数，而不是算法自己学习出来的。
    就像做饭时的"火候大小"和"盐的多少"，需要厨师（我们）来决定。
    """

    # === 环境相关 ===
    ENV_NAME = "CartPole-v1"    # 环境名称，CartPole 是小车倒立摆任务

    # === 训练相关 ===
    EPISODES = 5000             # 总共训练多少局（一轮游戏叫一个 episode）
    MAX_STEPS = 500             # 每局最多走多少步

    # === DQN 核心参数 ===
    GAMMA = 0.99                # 折扣因子 (γ)，范围 0~1
    """
    折扣因子解释：
    -------------
    γ = 0.99 意味着"未来的奖励在现在看来值多少钱"

    举个例子：
    - 如果 γ = 0.99，明天的 1 分 ≈ 今天的 0.99 分
    - 如果 γ = 0.5，明天的 1 分 ≈ 今天的 0.5 分

    γ 越大，越看重长期奖励（有远见）
    γ 越小，越看重短期奖励（短视）
    """

    LEARNING_RATE = 0.001       # 学习率 (α)
    """
    学习率解释：
    -------------
    每次更新网络参数时，"迈多大步子"

    学习率太大 → 步子太大，可能跨过最优解，训练不稳定
    学习率太小 → 步子太小，学习太慢，要训练很久

    0.001 是一个比较常用的值
    """

    BATCH_SIZE = 128            # 批次大小
    """
    批次大小解释：
    -------------
    每次训练神经网络时，用多少个样本

    BATCH_SIZE = 128 表示每次从经验池里随机抽取 128 条经验来学习

    太小（比如 8）：训练不稳定，像只看了几个例子就下结论
    太大（比如 1024）：内存占用大，训练慢
    """

    # === 经验回放相关 ===
    MEMORY_SIZE = 10000         # 经验池容量
    """
    经验池容量解释：
    ---------------
    能存储多少条"经验"（就是过去的游戏经历）

    每条经验包含：(状态, 动作, 奖励, 下一状态, 是否结束)

    10000 表示能记住最近 10000 步的经历
    """

    # === 探索相关 ===
    EPSILON_START = 1.0         # 初始探索率
    """
    初始探索率解释：
    ---------------
    ε = 1.0 表示 100% 随机探索

    一开始我们什么都不懂，所以完全随机尝试，看看会发生什么
    """

    EPSILON_END = 0.05          # 最终探索率
    """
    最终探索率解释：
    ---------------
    ε = 0.05 表示 5% 随机探索

    训练很久之后，我们基本学会了，但仍然保留 5% 的随机探索
    这样偶尔还能发现新策略
    """

    EPSILON_DECAY = 0.9999      # 探索率衰减
    """
    探索率衰减解释：
    ---------------
    每训练一次，ε = ε × 0.9999

    例子：
    - 开始：ε = 1.0（完全随机）
    - 100 次后：ε ≈ 0.99
    - 1000 次后：ε ≈ 0.90
    - 5000 次后：ε ≈ 0.60
    - 慢慢降到 0.05

    这就像：小时候什么都想试试（探索）
    长大后有了经验，大部分时候按经验办事，但偶尔也尝试新事物
    """


# ============================================================
# 经验回放缓冲区 - 存储和采样游戏经历
# ============================================================

class ReplayBuffer:
    """
    经验回放缓冲区

    通俗解释：
    ---------
    想象你在学玩游戏：
    - 每走一步，你就在笔记本上记下来："在这种情况做了这个动作，得了多少分"
    - 这个笔记本就是"经验池"
    - 训练时，从笔记本里随机翻几页来学习，而不是按顺序看

    为什么要随机？
    -------------
    如果按顺序学，相邻的经历太像了，会"钻牛角尖"
    随机抽取可以学到更多样化的经验
    """

    def __init__(self, capacity: int):
        """
        初始化经验池

        Args:
            capacity: 能存多少条经验（就像笔记本能写多少页）
        """
        # deque 是双端队列，这里用作"固定大小的容器"
        # 超出容量时，最老的经验会被自动删除
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存入一条经验

        Args:
            state: 当前状态（比如：杆子向左歪）
            action: 做了什么动作（比如：向左推）
            reward: 获得多少奖励（比如：+1 分）
            next_state: 之后变成什么状态（比如：杆子正回来了）
            done: 游戏是否结束（True/False）
        """
        # 把经验打包成元组，存入 buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        随机抽取一批经验

        Args:
            batch_size: 要抽取多少条

        Returns:
            五个数组：状态数组、动作数组、奖励数组、下一状态数组、结束数组
        """
        # 从 buffer 中随机抽取 batch_size 条经验
        batch = random.sample(self.buffer, batch_size)

        # 把打包的数据拆开
        # zip(*batch) 就像把一堆 (a,b,c) 变成 (a,a,a...), (b,b,b...), (c,c,c...)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),       # 转成 numpy 数组，方便计算
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        """返回当前存储了多少条经验"""
        return len(self.buffer)


# ============================================================
# Q 网络 - 神经网络，负责预测每个动作的价值
# ============================================================

class QNetwork(nn.Module):
    """
    Q 网络

    通俗解释：
    ---------
    这是一个"预测器"，输入当前状态，输出每个动作的"价值"

    对于 CartPole：
    - 输入：[小车位置, 小车速度, 杆子角度, 杆子角速度]（4 个数字）
    - 输出：[向左推的价值, 向右推的价值]（2 个数字）

    价值越高，表示这个动作越"好"
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化 Q 网络

        Args:
            state_dim: 状态有几个数字（CartPole 是 4）
            action_dim: 有几个动作可选（CartPole 是 2）
            hidden_dim: 隐藏层大小（网络的"脑容量"）
        """
        # 先调用父类的初始化（这是 PyTorch 的要求）
        super(QNetwork, self).__init__()

        # 用 nn.Sequential 构建一个"层叠层"的网络
        self.network = nn.Sequential(
            # === 第一层：输入 → 隐藏层 ===
            nn.Linear(state_dim, hidden_dim),  # 全连接层：4 → 128
            nn.ReLU(),                         # 激活函数：引入非线性

            # === 第二层：隐藏层 → 隐藏层 ===
            nn.Linear(hidden_dim, hidden_dim), # 全连接层：128 → 128
            nn.ReLU(),                         # 激活函数

            # === 第三层：隐藏层 → 输出 ===
            nn.Linear(hidden_dim, action_dim), # 全连接层：128 → 2
            # 注意：这里没有激活函数！
            # 因为 Q 值可以是任意实数（正的、负的都可以）
        )

    def forward(self, state):
        """
        前向传播：输入状态，输出 Q 值

        Args:
            state: 状态张量，shape [batch_size, state_dim]
                   比如 [32, 4] 表示 32 个状态，每个 4 维

        Returns:
            Q 值，shape [batch_size, action_dim]
            比如 [32, 2] 表示 32 个状态，每个对应 2 个动作的 Q 值
        """
        return self.network(state)


# ============================================================
# DQN 智能体 - 做决策和学习的主体
# ============================================================

class DQNAgent:
    """
    DQN 智能体

    通俗解释：
    ---------
    这就像一个"游戏玩家"，它：
    1. 观察游戏状态
    2. 决定做什么动作
    3. 记住经历
    4. 不断学习，变得越来越强
    """

    def __init__(self, state_dim: int, action_dim: int, config: Config = None):
        """
        初始化 DQN 智能体

        Args:
            state_dim: 状态维度（CartPole 是 4）--就是有几个状态
            action_dim: 动作数量（CartPole 是 2）--就是有几个动作（本case向左向右两个动作）
            config: 配置对象，如果没提供就用默认的
        """

        # === 处理配置 ===
        # 如果调用者没有提供 config，就用默认的 Config()
        if config is None:
            config = Config()

        # 保存配置和维度信息
        self.state_dim = state_dim          # 比如是 4
        self.action_dim = action_dim        # 比如是 2
        self.config = config                # 保存配置对象

        # === 创建神经网络 ===
        # Q 网络：用来做决策和学习的网络
        self.q_network = QNetwork(state_dim, action_dim)
        # 这是一个神经网络，输入状态，输出每个动作的 Q 值

        # 目标网络：用来计算目标的"参考"网络
        self.target_network = QNetwork(state_dim, action_dim)
        # 目标网络的参数从 Q 网络复制过来
        self.target_network.load_state_dict(self.q_network.state_dict())
        # state_dict() 存储了网络的所有参数

        # 为什么要两个网络？
        # 如果只有一个网络，目标和预测都在变化，训练会不稳定
        # 就像打靶时，靶子不动比靶子一直移动更容易瞄准

        # === 创建优化器 ===
        # 优化器负责更新网络的参数
        self.optimizer = optim.Adam(
            self.q_network.parameters(),    # 要更新哪些参数
            lr=config.LEARNING_RATE         # 学习率
        )
        # Adam 是最常用的优化器之一，它自适应地调整学习率

        # === 创建经验池 ===
        self.replay_buffer = ReplayBuffer(config.MEMORY_SIZE)
        # 用来存储游戏经历

        # === 初始化探索率 ===
        self.epsilon = config.EPSILON_START
        # 一开始完全随机探索（ε = 1.0）

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-贪婪策略）

        通俗解释：
        ---------
        ε-贪婪策略 = 随机探索 + 利用经验

        - 有 ε 的概率：随机选一个动作（探索未知）
        - 有 1-ε 的概率：选 Q 值最大的动作（利用已知）

        这就像：
        - 有时候去一家从没去过的餐厅探索
        - 有时候去熟悉的餐厅（因为你知道它好吃）

        Args:
            state: 当前状态，numpy 数组
            training: 是否在训练模式
                     - True：使用 ε-贪婪策略
                     - False：只用最优策略（不探索）

        Returns:
            选择的动作（整数，比如 0 或 1）
        """

        # === 情况 1：探索阶段 ===
        # 如果在训练中，且随机数小于 ε，则随机选择动作
        if training and random.random() < self.epsilon:
            # random.random() 返回 0~1 之间的随机数
            # 比如 ε = 0.5，那么有 50% 的概率走这个分支

            # random.randrange(n) 返回 0 到 n-1 之间的随机整数
            return random.randrange(self.action_dim)
            # 如果 action_dim = 2，则返回 0 或 1

        # === 情况 2：利用阶段 ===
        # 使用神经网络预测 Q 值，选择最大的

        # 1. 把 numpy 数组转换成 PyTorch 张量
        state = torch.FloatTensor(state)
        # FloatTen sor 是 PyTorch 的浮点数类型

        # 2. 增加一个 batch 维度
        state = state.unsqueeze(0)
        # 原本 shape 是 [4]，现在变成 [1, 4]
        # 因为神经网络期望输入是 [batch_size, state_dim]

        # 3. 计算 Q 值（不计算梯度，因为只是预测）
        with torch.no_grad():
            # torch.no_grad() 表示不需要计算梯度
            # 这样更快，因为不需要反向传播
            q_values = self.q_network(state)
            # q_values.shape 是 [1, 2]
            # 比如 [[0.5, 1.2]]，表示向左价值 0.5，向右价值 1.2

        # 4. 选择 Q 值最大的动作
        action = q_values.argmax(1).item()
        # argmax(1) 在第 1 维度（action_dim）上找最大值的索引
        # 比如 [[0.5, 1.2]] 的 argmax 是 1
        # .item() 把张量转换成 Python 普通数字

        return action
        # 返回 0 或 1

    def store_experience(self, state, action, reward, next_state, done):
        """
        存储一条经验到回放缓冲区

        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        # 直接调用经验池的 push 方法

    def train(self):
        """
        训练 Q 网络（一次更新步骤）

        通俗解释：
        ---------
        这就是"学习"的过程：

        1. 从经验池随机抽一批经历
        2. 计算"目标值"（应该得到的 Q 值）
        3. 计算"预测值"（当前网络预测的 Q 值）
        4. 计算差距（损失），调整网络参数让差距变小

        DQN 的核心公式：
        --------------
        目标 Q 值 = 奖励 + γ × 下一状态的最大 Q 值

        损失 = (预测 Q 值 - 目标 Q 值)²

        Returns:
            loss: 损失值（如果经验池不够则返回 None）
        """

        # 如果经验池里的经验不够，不训练
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None

        # === 第 1 步：从经验池采样 ===
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.BATCH_SIZE
        )
        # 每个都是 numpy 数组，shape [batch_size, ...]

        # === 第 2 步：转换成 PyTorch 张量 ===
        states = torch.FloatTensor(states)      # [batch_size, state_dim]
        actions = torch.LongTensor(actions)     # [batch_size] 用整数类型
        rewards = torch.FloatTensor(rewards)    # [batch_size]
        next_states = torch.FloatTensor(next_states)  # [batch_size, state_dim]
        dones = torch.FloatTensor(dones)        # [batch_size]

        # === 第 3 步：计算目标 Q 值 ===
        # Q_target = reward + γ × max Q_target(next_state)

        with torch.no_grad():  # 目标网络不需要梯度
            # 3.1 用目标网络计算下一状态的 Q 值
            next_q_values = self.target_network(next_states)
            # shape: [batch_size, action_dim]

            # 3.2 取每个状态的最大 Q 值
            next_q_max = next_q_values.max(1)[0]
            # max(1)[0] 在 action_dim 维度上取最大值，返回数值而非索引
            # shape: [batch_size]

            # 3.3 计算目标 Q 值
            # 如果 done=1（游戏结束），就没有未来奖励，只有当前奖励
            # 如果 done=0（游戏继续），有当前奖励 + 打折的未来奖励
            target_q_values = rewards + self.config.GAMMA * next_q_max * (1 - dones)
            # shape: [batch_size]

        # === 第 4 步：计算当前 Q 值 ===
        # 我们需要的是"在状态 s，选择动作 a 的 Q 值"
        # 但 q_network 给出的是所有动作的 Q 值

        # 4.1 计算所有动作的 Q 值
        current_q_values = self.q_network(states)
        # shape: [batch_size, action_dim]

        # 4.2 用 gather 操作提取实际采取的动作的 Q 值
        action_indices = actions.unsqueeze(1)  # [batch_size] → [batch_size, 1]
        q_values = current_q_values.gather(1, action_indices).squeeze(1)
        # gather 沿指定维度收集元素
        # 结果 shape: [batch_size]

        # === 第 5 步：计算损失并更新 ===
        # 使用 Smooth L1 Loss（比普通 MSE 更鲁棒）
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        # 反向传播更新参数
        self.optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()             # 计算梯度
        self.optimizer.step()       # 更新参数

        # === 第 6 步：更新探索率 ===
        # ε 慢慢变小，逐渐从探索转向利用
        self.epsilon = max(
            self.config.EPSILON_END,           # 最小值
            self.epsilon * self.config.EPSILON_DECAY  # 每次乘衰减系数
        )

        return loss.item()  # 返回损失值（Python 标量）

    def update_target_network(self):
        """
        将主网络的参数复制到目标网络

        通俗解释：
        ---------
        定期"同步"目标网络，让目标网络跟上主网络的进度

        这就像：
        - 主网络是"正在学习的学生"
        - 目标网络是"参考答案"
        - 每隔一段时间，更新参考答案

        为什么要这样做？
        ---------------
        如果目标和预测都在变化，就像"追着一辆移动的车"，
        目标固定更容易训练
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        # load_state_dict() 加载参数
        # state_dict() 获取参数

    def save(self, path: str):
        """
        保存模型到文件

        Args:
            path: 保存路径，比如 "models/checkpoint.pth"
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """
        从文件加载模型

        Args:
            path: 模型文件路径
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
