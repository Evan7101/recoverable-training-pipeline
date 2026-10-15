# Recoverable Training Pipeline

可恢复训练流水线实验：演示可恢复的 checkpoint/resume，并判断在状态不完整时哪些恢复保证是诚实且有价值的。

## 目录结构

```
recovery-baseline/
├── README.md                 # 本文件
├── .gitignore
├── docs/
│   ├── PLAN.md               # 工作规划与验收记录
│   ├── RISK-ASSUMPTION.md    # 风险最高的假设：识别、最小检查与结论
│   ├── DECISIONS.md          # 四个决策问题的个人回答
│   └── REPORT.md             # 评估报告：结果汇总与未验证风险登记
├── brief/
│   ├── recovery-brief.md     # 任务书
│   └── recovery-brief-zh.md  # 任务书（中文版）
├── toy/                      # 机制层：1 维 toy，纯 Python 标准库
│   ├── recovery-baseline.py  # 任务书自带基线（写入 cwd 的 checkpoint.json，未改动）
│   ├── check_riskiest_assumption.py
│   ├── train_recoverable.py
│   ├── checkpoint.json       # check_riskiest_assumption.py 生成的 checkpoint
│   └── results/              # 机制层运行记录
└── mlp/                      # 真实模型层：PyTorch MLP + MNIST
    ├── train_mlp.py
    ├── results/              # 真实模型层运行记录（equivalence.md、logs/、checkpoints/ 等）
    └── data/                 # MNIST 数据集（不入库，.gitignore）
```

## 交付物（三份独立文档）

| 文档 | 内容 |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | 工作规划：实现/阅读/评估/沟通四类任务、优先级与依赖、AI 作业审计、后续行动、验收记录，跨 5 个工作会话更新 |
| [`docs/RISK-ASSUMPTION.md`](docs/RISK-ASSUMPTION.md) | 风险最高的假设：识别理由、最小但有效的检查、实测结果与结论理由 |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 四个决策问题的个人回答：优先级与价值取向、利益方权衡、不确定性下的保证承诺、责任与反思 |

支撑材料：[`docs/REPORT.md`](docs/REPORT.md)（结果汇总与未验证风险登记）、`toy/results/`（机制层运行记录）、`mlp/results/`（真实模型层运行记录）。

## 两层实验

- **机制层**（`toy/train_recoverable.py`，纯标准库）：1 维 toy 上的位级等价证明。全状态恢复在切分点 1–9 全部位级一致（含 RNG 覆盖测试）；缺失/损坏/不兼容 checkpoint 显式失败（5 例负例）。
- **真实模型层**（`mlp/train_mlp.py`，PyTorch）：MLP + MNIST 三档恢复对比——全状态恢复容差内一致且指标无差异；丢优化器状态 / 丢 RNG 状态的部分恢复实测改变最终精度（\|Δacc\| 0.8–2.0%）。

## 运行

所有命令均在仓库根目录（`recovery-baseline/`）下执行；脚本内输出路径均解析到脚本自身所在目录，与当前工作目录无关。

```bash
# 机制层（Python 3 标准库即可）
python3 toy/check_riskiest_assumption.py
python3 toy/train_recoverable.py compare --total 10 --splits 1 2 3 4 5 6 7 8 9
python3 toy/train_recoverable.py selftest

# 真实模型层（需要 torch/torchvision；`python` 用任意含两者的环境，
# 本集群为共享 anaconda3 环境的 pytorch_gpu_2.0.1；
# MNIST 首次运行时自动下载到 mlp/data/）
python mlp/train_mlp.py compare --epochs 3 --split-epoch 1 --split-batch 47
python mlp/train_mlp.py selftest
```

## 说明

- `toy/recovery-baseline.py` 是任务书自带基线，保持原样（含其 cwd 相对行为）：它在**当前工作目录**写入 `checkpoint.json`，从仓库根目录运行会生成根目录的 `checkpoint.json`；`toy/checkpoint.json` 是 `toy/check_riskiest_assumption.py` 生成的。
- `mlp/data/`（MNIST，约 11MB）不入库，首次运行自动从 torchvision 默认镜像（ossci S3）下载。
- 实现工作有 AI 辅助；规划、验收、风险判断与全部决策文档为负责人亲笔（详见各文档披露声明）。

## 最终交付

- **交付修改**：任务基线扩展为两层实验——机制层 toy（全状态 checkpoint、切分点 1–9 位级等价验证、5 例显式失败负例）与真实模型层 PyTorch MLP（全状态 / 丢优化器 / 丢 RNG 三档恢复对比）；三份交付文档 `docs/PLAN.md`、`docs/RISK-ASSUMPTION.md`、`docs/DECISIONS.md`；支撑证据 `docs/REPORT.md` 与全部运行记录入库；目录重组为 `docs/ brief/ toy/ mlp/`（git mv 保留历史）。
- **运行说明**：见上文「运行」一节——机制层仅需 Python 3 标准库；MLP 层需 torch/torchvision（集群复用 `pytorch_gpu_2.0.1` 环境），MNIST 首次运行自动下载到 `mlp/data/`。
- **结果**：机制层全状态恢复在切分点 1–9 逐步位级一致（含 RNG 覆盖测试）；MLP 层全状态恢复默认容差内一致、最终精度无差异（0.882，epoch 边界与中周期两个切分点复现）；两类部分恢复实测改变最终精度（|Δacc| 0.8–2.0%）；两层共 10 例负例全部显式失败（exit=2、可读报错）；全部关键结果经负责人独立复跑验证（含输出哈希比对）。
- **未解决问题**：见 `docs/REPORT.md` §5A 未验证风险登记——最关键三条：语义级损坏（语法合法但值错误）当前不可检、GPU/多线程场景仅文献级证据、部分恢复偏差量级不可跨规模外推。
- **后续行动**：P0/P1 事项已全部关闭，无阻塞遗留。可选改进：为 checkpoint 增加校验和以关闭语义级损坏风险；GPU/多线程环境下的等价性实测；真实规模下的部分恢复偏差收敛研究。
