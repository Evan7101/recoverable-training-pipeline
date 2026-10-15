# Recoverable Training Pipeline

可恢复训练流水线实验：演示可恢复的 checkpoint/resume，并判断在状态不完整时哪些恢复保证是诚实且有价值的。

## 交付物（三份独立文档）

| 文档 | 内容 |
|---|---|
| [`PLAN.md`](PLAN.md) | 工作规划：实现/阅读/评估/沟通四类任务、优先级与依赖、AI 作业审计、后续行动、验收记录，跨 5 个工作会话更新 |
| [`RISK-ASSUMPTION.md`](RISK-ASSUMPTION.md) | 风险最高的假设：识别理由、最小但有效的检查、实测结果与结论理由 |
| [`DECISIONS.md`](DECISIONS.md) | 四个决策问题的个人回答：优先级与价值取向、利益方权衡、不确定性下的保证承诺、责任与反思 |

支撑材料：[`REPORT.md`](REPORT.md)（结果汇总与未验证风险登记）、`results/`（机制层运行记录）、`results_mlp/`（真实模型层运行记录）。

## 两层实验

- **机制层**（`train_recoverable.py`，纯标准库）：1 维 toy 上的位级等价证明。全状态恢复在切分点 1–9 全部位级一致（含 RNG 覆盖测试）；缺失/损坏/不兼容 checkpoint 显式失败（5 例负例）。
- **真实模型层**（`train_mlp.py`，PyTorch）：MLP + MNIST 三档恢复对比——全状态恢复容差内一致且指标无差异；丢优化器状态 / 丢 RNG 状态的部分恢复实测改变最终精度（\|Δacc\| 0.8–2.0%）。

## 运行

```bash
# 机制层（Python 3 标准库即可）
python3 train_recoverable.py compare --total 10 --splits 1 2 3 4 5 6 7 8 9
python3 train_recoverable.py selftest
python3 check_riskiest_assumption.py

# 真实模型层（需要 torch/torchvision；MNIST 首次运行时自动下载到 data_mlp/）
python train_mlp.py compare --epochs 3 --split-epoch 1 --split-batch 47
python train_mlp.py selftest
```

## 说明

- `data_mlp/`（MNIST，约 11MB）不入库，首次运行自动从 torchvision 默认镜像（ossci S3）下载。
- 实现工作有 AI 辅助；规划、验收、风险判断与全部决策文档为负责人亲笔（详见各文档披露声明）。
