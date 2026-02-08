# 🏛️ Prometheus - 强化学习框架

> 为人类带来火种的 RL 框架

## 📖 这个项目是什么？

Prometheus 是一个**从零开始**打造的强化学习框架，目的是通过实际动手来学习 AI Infrastructure。

## 🎯 学习目标

- 理解强化学习的核心原理
- 学习如何设计高效的 AI 框架
- 掌握 PyTorch 和系统编程的结合
- 最终实现一个可用的 RL 框架

## 📁 项目结构

```
Prometheus/
├── prometheus/          # 框架核心代码
│   ├── __init__.py
│   ├── core.py              # Config, ReplayBuffer
│   ├── envs/                # 环境模块 ✨ v0.1.0
│   │   ├── base.py          # 环境抽象接口
│   │   └── gym_wrapper.py   # Gym 环境包装器
│   ├── policies/            # 策略模块 ✨ v0.1.0
│   │   ├── base.py          # 策略抽象接口
│   │   └── dqn.py           # DQN 策略
│   ├── agents/              # 智能体模块 ✨ v0.1.0
│   │   ├── base.py          # 智能体抽象接口
│   │   └── dqn.py           # DQN 智能体
│   └── trainers/            # 训练器模块 ✨ v0.1.0
│       ├── base.py          # 训练器抽象接口
│       └── dqn.py           # DQN 训练器
├── examples/            # 示例代码
│   ├── 01_cartpole_dqn.py   # DQN 训练示例（v0.0.1）
│   ├── 02_watch_agent.py    # 观看智能体表现
│   └── 03_new_framework.py  # 使用新框架（v0.1.0）✨
├── docs/                # 学习笔记
│   ├── plan.md             # 学习规划
│   └── 学习笔记_01_火种篇.md
├── tests/               # 测试代码
├── venv/                # 虚拟环境
├── run.sh               # 便捷运行脚本
└── requirements.txt     # 依赖列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 虚拟环境已创建，直接激活即可
source venv/bin/activate

# 或者在 Windows 上
venv\Scripts\activate
```

### 2. 运行示例

**方式一：使用 run.sh 脚本（推荐）**
```bash
./run.sh examples/03_new_framework.py
```

**方式二：直接运行（需要设置 PYTHONPATH）**
```bash
source venv/bin/activate
PYTHONPATH=. python examples/03_new_framework.py
```

## 📚 示例说明

### 03_new_framework.py - 新框架使用 ✨ 推荐

展示 Prometheus v0.1.0 新框架的使用方式。

**运行**：
```bash
./run.sh examples/03_new_framework.py
```

**代码示例**：
```python
from prometheus.envs import make_gym_env
from prometheus.agents import DQNAgent
from prometheus.trainers import DQNTrainer, TrainerConfig

# 创建环境
env = make_gym_env("CartPole-v1")

# 创建智能体
agent = DQNAgent(state_dim=4, action_dim=2)

# 创建训练器
config = TrainerConfig(max_episodes=500)
trainer = DQNTrainer(config=config)

# 开始训练
trainer.train(env, agent)
```

### 01_cartpole_dqn.py - DQN 训练

强化学习的 "Hello World"，用 DQN 算法解决 CartPole 问题。

**运行**：
```bash
./run.sh examples/01_cartpole_dqn.py
```

### 02_watch_agent.py - 观看智能体

快速训练一个模型，然后打开窗口观看智能体如何平衡杆子。

**运行**：
```bash
./run.sh examples/02_watch_agent.py
```

**注意**：需要图形界面支持。

## 📖 学习进度

- [x] 阶段一：火种 - 跑通第一个 RL 实验（v0.0.1）
- [x] 阶段二：铸炉 - 设计框架基础架构（v0.1.0）✨ 当前版本
- [ ] 阶段三：添柴 - 实现核心算法
- [ ] 阶段四：炼金 - 性能优化
- [ ] 阶段五：燎原 - 生态完善

## 🔧 框架 API

### 环境模块 (prometheus.envs)

```python
from prometheus.envs import make_gym_env

# 创建 Gym 环境
env = make_gym_env("CartPole-v1")
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)
```

### 智能体模块 (prometheus.agents)

```python
from prometheus.agents import DQNAgent

agent = DQNAgent(state_dim=4, action_dim=2)
action = agent.act(state)           # 选择动作
agent.remember(s, a, r, s2, done)   # 存储经验
metrics = agent.learn()             # 学习
```

### 训练器模块 (prometheus.trainers)

```python
from prometheus.trainers import DQNTrainer, TrainerConfig

config = TrainerConfig(
    max_episodes=1000,
    eval_interval=100,
    log_interval=10
)
trainer = DQNTrainer(config=config)
trainer.train(env, agent)
```

## 📖 学习资源

- [Spinning Up in RL (OpenAI)](https://spinningup.openai.com/)
- [DQN 论文](https://www.nature.com/articles/nature14236)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [PyTorch 教程](https://pytorch.org/tutorials/)

## 📝 版本历史

### v0.1.0 (铸炉版本) - 当前版本
- ✨ 模块化架构设计
- ✨ 环境模块 (envs)
- ✨ 策略模块 (policies)
- ✨ 智能体模块 (agents)
- ✨ 训练器模块 (trainers)
- ✨ 回调系统

### v0.0.1 (火种版本)
- DQN 算法实现
- CartPole 示例
- 基础训练循环
