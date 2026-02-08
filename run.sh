#!/bin/bash
# Prometheus 项目运行脚本

# 激活虚拟环境并设置 PYTHONPATH
source venv/bin/activate
export PYTHONPATH=.

# 运行指定的脚本
if [ -z "$1" ]; then
    echo "用法: ./run.sh <脚本路径>"
    echo ""
    echo "示例:"
    echo "  ./run.sh examples/01_cartpole_dqn.py    # 训练 DQN"
    echo "  ./run.sh examples/02_watch_agent.py     # 观看智能体"
    exit 1
fi

python "$@"
