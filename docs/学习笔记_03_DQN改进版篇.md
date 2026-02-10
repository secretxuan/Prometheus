# Prometheus 学习笔记 #03 - DQN 改进版篇

> 版本 v0.2.0 - DQN 改进版：Double, Dueling, PER, Rainbow

## 📖 阶段三目标（DQN 改进版）

在标准 DQN 基础上，实现四种改进算法。

---

## 一、Double DQN

### 核心问题：Q 值过高估计

**为什么 DQN 会过高估计 Q 值？**

标准 DQN 计算目标 Q 值时：
```python
# 目标网络既选动作又评估价值
next_q_values = target_network(next_states)
target_q = reward + gamma * next_q_values.max()
```

问题：`max` 操作会选到最大的 Q 值，但这个最大值可能是**噪声**导致的，不是真实价值。

### Double DQN 解决方案

**解耦动作选择和价值评估**：

```python
# 用主网络选择动作
next_action = policy_network(next_states).argmax()

# 用目标网络评估该动作的价值
next_q_values = target_network(next_states)
target_q = reward + gamma * next_q_values[next_action]
```

### 通俗解释

| 算法 | 就像... |
|------|---------|
| 标准 DQN | 用同一本教材出题和评分，容易"刷分" |
| Double DQN | 用不同教材，出题和评分分开，更客观 |

### 代码实现

```python
class DoubleDQNAgent(DQNAgentBase):
    def compute_target_q(self, next_states, rewards, dones):
        # Double DQN: 解耦选择和评估
        with torch.no_grad():
            # 主网络选择动作
            next_actions = self.policy.q_network(next_states).argmax(1)
            # 目标网络评估
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.config.GAMMA * next_q_values * (1 - dones)
        return target_q
```

---

## 二、Dueling DQN

### 核心思想：分解 Q 值

标准 DQN 直接学习 Q(s,a)，但很多情况下：

- **状态本身的好坏**（比如"快到终点了"）与**具体动作**关系不大
- 只需要评估"这是个好状态"，而不需要精确区分每个动作

### Q 值分解公式

```
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

其中：
- **V(s)**：状态价值，表示这个状态本身有多好
- **A(s,a)**：动作优势，表示这个动作比平均动作好多少

### 网络结构

```
        输入状态 s
            |
        共享特征层
            |
      +-----+-----+
      |           |
   Value Stream  Advantage Stream
   (状态价值 V)  (动作优势 A)
      |           |
      +-----+-----+
            |
    Q(s,a) = V + A - mean(A)
```

### 代码实现

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.shared_layer = nn.Sequential(...)  # 共享特征
        self.value_stream = nn.Sequential(...)   # V(s)
        self.advantage_stream = nn.Sequential(...)  # A(s,a)

    def forward(self, state):
        features = self.shared_layer(state)
        value = self.value_stream(features)       # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, action_dim]
        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

### 为什么减去 mean(A)？

为了保证可识别性：
- 如果所有 A(s,a) 都加上同一个常数 c，Q 值不变
- 减去 mean(A) 可以固定这个常数，使学习更稳定

---

## 三、优先级经验回放（PER）

### 核心问题：均匀采样效率低

标准 DQN 从经验池均匀随机采样，但：
- 有些经验很"普通"，学不学差别不大
- 有些经验很"意外"，值得多学习几次

### 解决方案：按优先级采样

**优先级 = |TD 误差| + ε**

TD 误差越大 → 预测越不准 → 越值得学习

### SumTree 数据结构

为了 O(log n) 采样，使用 SumTree：

```
            [p0+p1+p2+p3]  <- 根节点（总和）
               /      \
         [p0+p1]      [p2+p3]
          /  \         /  \
        p0    p1     p2    p3  <- 叶子（存储数据）
```

### 重要性采样权重

按优先级采样会改变数据分布，需要用权重修正：

```python
weight = (N * P(i))^(-beta)
```

- beta 从 0.4 线性增长到 1
- 越常用的经验，权重越小（防止过拟合）

### 代码实现

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4):
        self.sum_tree = SumTree(capacity)
        self.alpha = alpha  # 优先级指数
        self.beta_start = beta_start

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + epsilon) ** alpha
            self.sum_tree.update(idx, priority)
```

---

## 四、Rainbow DQN

### 整合所有改进

Rainbow = Double + Dueling + PER

| 改进 | 解决的问题 |
|------|-----------|
| Double DQN | Q 值过高估计 |
| Dueling DQN | 状态价值与动作优势分离 |
| PER | 提高学习效率 |

### 代码结构

```python
class RainbowAgent:
    def __init__(...):
        # Dueling 网络
        self.policy = RainbowDQNPolicy(...)
        self.target_network = DuelingQNetwork(...)

        # PER 缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(...)

    def learn(self):
        # PER 采样（返回索引和权重）
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(batch_size)

        # Double DQN 计算目标
        next_actions = self.policy.q_network(next_states).argmax(1)
        next_q_target = self.target_network(next_states)
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * next_q_values * (1 - dones)

        # 加权损失
        td_errors = torch.abs(target_q - q_values)
        loss = (weights * loss_fn(q_values, target_q)).mean()

        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors)
```

---

## 五、算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| DQN | 简单稳定 | Q 值过高估计 | 入门学习 |
| Double DQN | 估值更准确 | 增加计算量 | 通用场景 |
| Dueling DQN | 更好学习状态价值 | 网络更复杂 | 动作价值相近的状态 |
| PER | 学习效率高 | 实现复杂 | 经验质量差异大 |
| Rainbow | 性能最好 | 实现最复杂 | 追求最佳性能 |

---

## 六、实验结果建议

在 CartPole-v1 上的预期表现：

| 算法 | 平均得分 | 收敛速度 |
|------|----------|----------|
| DQN | ~400 | 中等 |
| Double DQN | ~420 | 稍快 |
| Dueling DQN | ~430 | 中等 |
| PER | ~450 | 较快 |
| Rainbow | ~470+ | 最快 |

---

## 七、实现要点

### 1. 向后兼容性

重构后保持原有导入路径有效：

```python
# 旧代码仍然有效
from prometheus.agents import DQNAgent
from prometheus.policies import DQNPolicy

# 新代码推荐
from prometheus.agents.dqn import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PERAgent, RainbowAgent
```

### 2. 模块化设计

```
prometheus/
├── policies/dqn/
│   ├── base.py      # DQN 基类
│   ├── double.py    # Double DQN
│   ├── dueling.py   # Dueling DQN
│   └── rainbow.py   # Rainbow
├── agents/dqn/
│   ├── base.py      # DQN 基类
│   ├── double.py
│   ├── dueling.py
│   ├── per.py
│   └── rainbow.py
└── core.py          # SumTree, PrioritizedReplayBuffer
```

---

## 📝 今日总结

### 学到的知识：
1. **Double DQN**：解耦动作选择和评估
2. **Dueling DQN**：Q = V + A 分解
3. **PER**：优先级采样 + 重要性采样权重
4. **SumTree**：O(log n) 采样和更新
5. **Rainbow**：整合多种改进

### 框架变化：
- 新增 `policies/dqn/` 子模块
- 新增 `agents/dqn/` 子模块
- 新增 `SumTree` 和 `PrioritizedReplayBuffer`
- 重构 DQN 代码结构

---

*下一步：策略梯度方法（REINFORCE, A2C, PPO）*
