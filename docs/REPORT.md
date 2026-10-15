# 评估报告：可恢复训练流水线（E1）

- 日期：2026-10-15
- 范围：小型 checkpoint/resume 实验，CPU，纯 Python 标准库
- 工作方式：实现由 AI 子代理（coder）在负责人验收下完成；调研由只读子代理完成；本报告与决策日志为负责人亲笔。AI 辅助已按任务书要求披露。

## 1. 运行命令（全部可复现）

以下命令在仓库根目录（`recovery-baseline/`）执行；机制层用系统 `python3`，真实模型层的 `python` 需为含 torch/torchvision 的解释器（本集群复用共享 anaconda3 环境的 `pytorch_gpu_2.0.1`，其他环境 `pip install torch torchvision` 亦可）。

```bash
python3 toy/recovery-baseline.py                                   # 基线（写入 cwd 的 checkpoint.json）
python3 toy/check_riskiest_assumption.py                           # 风险假设最小检查
python3 toy/train_recoverable.py run --total 10                    # 不间断运行
python3 toy/train_recoverable.py run --total 4 --save-checkpoint toy/results/manual_ckpt.json   # 中断并保存
python3 toy/train_recoverable.py run --total 10 --checkpoint toy/results/manual_ckpt.json       # 恢复
python3 toy/train_recoverable.py compare --total 10 --splits 4 7   # 等价性对比
python3 toy/train_recoverable.py compare --total 10 --splits 1 2 3 4 5 6 7 8 9  # 全切分点扫掠
python3 toy/train_recoverable.py selftest                          # 负例测试
python mlp/train_mlp.py compare --epochs 3 --split-epoch 1 --split-batch 47
python mlp/train_mlp.py selftest
```

## 2. 模型与 checkpoint 设计

- 训练模型：对 (w−3)² 做带动量的梯度下降（mu=0.9, lr=0.2），每步从全局 `random` 模块抽取消噪（σ=0.05）。动量速度 = 优化器状态；随机流 = 必须在恢复时还原的状态。
- checkpoint（JSON，临时文件 + `os.replace` 原子写入，fsync）：`format_version`、`step`、`weight`、`velocity`、`rng_state`（`random.getstate()` 忠实序列化，含 `gauss_next`）、`total_steps`。
- 关键正确性约束：每个训练步恰好消耗一次 RNG 且存/载 checkpoint 时不消耗 RNG——存/载期间的额外 RNG 消耗是恢复失配的经典原因。

## 3. 结果

### 3.1 等价性对比（`compare`，total=10，切分点 1–9 全覆盖）

| 场景 | 切分点 | 结果 |
|---|---|---|
| 全状态恢复（含 rng_state；恢复前全局 RNG 已被 `seed(999)` 覆盖） | 1–9（全部 9 个） | **BIT-EXACT: True ×9**——每个切分点 10/10 步逐步一致 |
| 部分恢复（不恢复 rng_state，独立重播种） | 1–9（全部 9 个） | 每个切分点均可测发散，最大逐步权重偏差 0.00095377–0.0240948 |

切分点 4、7 的早期结果（0.0240948 / 0.00893984）与全扫掠一致。完整逐步表格见 `toy/results/equivalence.md`。RNG 覆盖测试说明：恢复不是"碰巧撞上同一随机序列"，而是从保存的 `rng_state` 真实还原。

### 3.2 负例测试（`selftest`，5/5 通过）

| 案例 | exit code | stderr（摘录） |
|---|---|---|
| checkpoint 文件缺失 | 2 | `ERROR: checkpoint file not found: … (refusing to start from scratch)` |
| format_version 不兼容 | 2 | `ERROR: unsupported format_version 999 …` |
| JSON 截断/损坏 | 2 | `ERROR: checkpoint … is corrupt/truncated JSON: …` |
| 缺必需字段（velocity） | 2 | `ERROR: checkpoint … missing required key(s): velocity` |
| 字段类型错误（step 为字符串） | 2 | `ERROR: checkpoint … field 'step' must be a non-negative int, got str: 'abc'` |

全部输出见 `toy/results/negative_tests.txt`。五个案例的 stderr 均可读且不含 Python traceback。另一处显式失败（不属于 checkpoint 损坏但属于不一致请求）：resume 时 `--total` 小于 checkpoint 的 step，报 `ERROR: requested total steps 2 is behind checkpoint step 4 (refusing to go backwards)`（exit=2）；`--total` 等于 checkpoint step 视为幂等 no-op，允许成功。行为承诺：缺失/不兼容/不一致时显式失败，**绝不静默继续或静默初始化**。

### 3.3 失败案例

机制层：恢复轨迹的逐步对比最初只记录切分后的步，导致对比表错位——通过合并切分前段轨迹修复。真实模型层开发中另发现并修复两个真实 bug（详见 `mlp/results/run_log.md`）：(a) 中周期停止时 epoch 字段提前推进、`scheduler.step()` 提前调用，造成 epoch/step 不一致；(b) torch 2.0.1 的 `RandomSampler` 每轮迭代重新 `randperm`，naive 中周期续跑会拿到**下一轮**的数据排列（逐步损失差立即达 0.44）——通过对 shuffle generator 做 epoch 起点快照并在中周期停止时回绕修复。后者表明"DataLoader 采样状态"不是一句口号，而是有真实陷阱的恢复面。

### 3.4 真实模型层结果（T5，容差级主张）

实验：MLP 784→128→10，MNIST 子集（2000 训练 / 500 测试），SGD+momentum + StepLR，3 epoch（96 步），单线程 CPU + `torch.use_deterministic_algorithms(True)`（未触发算子报错），torch 2.0.1（复用集群环境，零安装）。checkpoint 含 model/optimizer/scheduler 三个 state_dict、epoch/step、torch/numpy/python 三处 RNG、DataLoader shuffle generator 状态。参考最终测试精度 0.882。每档恢复前全局 RNG 被刻意覆盖以证明还原真实生效；预分段一致性是逐位验证而非假设。

| 切分点 | 恢复档 | max 参数差 | allclose(默认容差) | 位级一致（仅观察） | 最终精度 | Δacc |
|---|---|---|---|---|---|---|
| epoch 1 末（第 32 步） | 全状态 | 0.0 | True | True | 0.882 | 0 |
| 同上 | 丢优化器/调度器状态 | 0.0358 | False | False | 0.870 | −0.012 |
| 同上 | 丢全部 RNG 状态 | 0.0360 | False | False | 0.890 | +0.008 |
| 中周期 batch 47 | 全状态 | 0.0 | True | True | 0.882 | 0 |
| 同上 | 丢优化器/调度器状态 | 0.0265 | False | False | 0.862 | −0.020 |
| 同上 | 丢全部 RNG 状态 | 0.0294 | False | False | 0.896 | +0.014 |

要点：
- 全状态恢复在默认容差下一致、指标无差异；位级一致被观察到，但按预写边界仅作观察——单线程 CPU + 全确定性开关 + 四源 RNG 全纳入是受控特例（R1 调研已说明不具外推性）。
- 两类部分恢复都改变最终指标（|Δacc| 0.8–2.0%）——"部分恢复必须披露"从 toy 机制结论升级为真实模型层的实测结论。
- 负例 5/5 显式失败（缺失/损坏/版本不兼容/缺键/类型错），语义与机制层一致；倒退式 resume 拒绝、等步 resume 幂等。
- 数据：MNIST 经 torchvision 默认 ossci S3 镜像下载成功（yann.lecun.com 源已弃用），存于 `mlp/data/`；未使用合成数据回退。
- 负责人独立复跑 `compare`：exit=0，`mlp/results/equivalence.md` 字节级一致（md5 相同）。
- 完整表格与逐 epoch 损失见 `mlp/results/equivalence.md`。

## 4. 已验证的保证 vs 未建立的保证

**已验证（本实验内，有运行证据）：**
1. 机制层：单进程、单 CPU、同软件版本下，全状态 checkpoint 恢复与不间断运行逐步位级一致（切分点 1–9 全扫掠 + RNG 覆盖测试）。
2. 机制层：丢失 RNG 状态的部分恢复会改变轨迹，偏差可测量（最大约 0.024）。
3. 缺失/损坏/不兼容/不一致 checkpoint 被显式拒绝，报错可读、退出码明确（机制层 5 例 + 倒退 resume；真实模型层 5 例，两层语义一致）。
4. 真实模型层（T5）：同环境 CPU 全状态恢复在默认浮点容差内一致（allclose=True），最终测试精度无差异（参考 0.882，epoch 边界与中周期两个切分点均复现）；两类部分恢复均实测改变最终精度（|Δacc| 0.8–2.0%）。位级一致被观察到，按预写边界仅作观察、不作保证。

**未建立（明确不外推）：**
1. GPU/生产环境：cuDNN benchmark 选路、TF32、非确定性原子加、cuBLAS workspace 都会使同代码两次运行不同（PyTorch 官方文档）。
2. 多线程 CPU 归约：浮点加法不可结合，线程调度改变归约顺序。
3. DataLoader 多 worker 的 RNG 派生与乱序语义不在本实验范围（T5 仅覆盖单 worker + shuffle generator 快照回绕的受控形式）。
4. 优化器状态仅覆盖带动量的 SGD；Adam 二阶矩等未覆盖。
5. 跨 Python/库版本的 `rng_state` 兼容性未验证。
6. T5 仅覆盖单线程 CPU、torch 2.0.1、MLP + MNIST 子集（2000/500）；真实规模、真实数据管道、跨版本均未建立。

诚实边界（引用 R1 调研，来源：PyTorch Reproducibility 文档等，见 `docs/PLAN.md` 第 2 节）：checkpoint 保证"状态无丢失地继续"，**不**保证"与从未中断的世界线逐位重合"；生产环境中最强可辩护主张是同硬件同软件下的容差内一致，跨硬件仅有统计等价。

> **保证分层（T5 预写，2026-10-15）**：本报告的主张分两层。**机制层（toy，`toy/train_recoverable.py`）**：受控环境下允许位级一致主张，证据为切分点 1–9 全扫掠。**真实模型层（T5，PyTorch MLP）**：无论实测结果如何，只允许"容差内一致 / 统计等价"主张，不允许位级主张——即便 CPU 全确定性开关下实测到位级一致，也仅作为有趣观察记录，不升级为保证。若实测在确定性开关全开下仍无法位级复现，将如实记录并解释（R1 调研预言此现象可能出现，属有价值的结果而非失败）。两层结论在 §3 与 T5 结果中分开陈述，不互相借用证据。

### 4.1 对照：本 toy 与 PyTorch 生产训练的保证（R1 调研汇总）

| 维度 | 本实验 toy | PyTorch 生产训练（官方文档语义） |
|---|---|---|
| checkpoint 内容 | weight、velocity、step、Python 全局 RNG 状态、total_steps、format_version | model `state_dict`、optimizer `state_dict`、epoch/loss、scheduler/AMP 状态、`torch`/`numpy`/CUDA 多方 RNG、数据采样位置；体积约为模型的 2–3 倍 |
| 优化器状态 | 仅动量一阶 | Adam 二阶矩等完整状态 |
| 位级等价的条件 | 已实测达成：单进程、单 CPU、同版本、9/9 切分点 | 需同硬件、同软件栈、确定性开关全开（`cudnn.deterministic`、`CUBLAS_WORKSPACE_CONFIG` 等），属昂贵特例（如 PaLM 的位级续训）而非常态 |
| 常态可辩护主张 | 受控环境内位级一致 | 同硬件同软件下容差内一致；跨硬件/跨版本仅统计等价 |
| 典型失效源 | 无（环境受控） | cuDNN benchmark 选路、TF32、非确定性原子加、多线程归约、DataLoader worker RNG/乱序、未纳入 checkpoint 的 RNG 消费者（dropout、NumPy） |
| 失败处理 | exit=2 显式报错，5 类负例实测 | 由框架与团队实践决定，无统一保证 |

来源：PyTorch Reproducibility 官方文档、PyTorch Saving/Loading 教程、NVIDIA TAO 确定性训练文档、`torch.utils.data` worker 种子语义、OpGuard（OSDI'26）、OPEN-1B（arXiv）。

## 5. 已知局限

- toy 模型维度为 1，无真实数据管道；结论用于说明机制，不用于估计真实训练中的偏差量级。
- `compare` 在单进程内完成，未测试跨进程/跨机恢复（JSON 往返已隐含跨进程语义）。
- 位级等价证据密度：2 个切分点；切分点 1–9 全覆盖可作为后续加强（见 `docs/PLAN.md` 后续行动）。
- 部分恢复的偏差大小依赖 toy 的噪声幅度 σ=0.05，不可外推到真实任务。

## 5A. 未验证风险登记（提交时仍开放）

1. **语义级损坏不可检**：两层 checkpoint 的"损坏检测"依赖解析失败（JSON/torch.load 抛错）。若文件被改成**语法合法但语义错误**的值（如 weight 被换成旧值、step 被改小），当前实现会照常恢复且不报警。未做校验和/哈希。
2. **GPU / 多线程 CPU / 多 worker DataLoader**：仅有文献级证据（R1 调研），无实测；T5 的位级观察明确不覆盖这些环境。
3. **跨版本兼容性**：Python `random`/`torch` 的 RNG 状态序列化格式跨版本未验证；checkpoint 的 `format_version=1` 尚无升级路径实测。
4. **优化器覆盖面**：仅 SGD+momentum（含 scheduler 计数）；Adam 二阶矩、AMP GradScaler 等未实现未测。
5. **规模外推**：T5 为 2000 样本 / 3 epoch；部分恢复的偏差量级（|Δacc| 0.8–2.0%）不可外推到真实训练。
6. **下游指标等价未闭环**：等价性检查覆盖训练轨迹与最终精度；下游任务（如用恢复后模型做选择/部署）的指标等价未验证。
7. **跨进程/跨机恢复**：T5 的跨进程手动恢复测过单一路径；跨节点文件系统、并发写 checkpoint 未测（原子写只防单机半写）。
8. **超参/数据变更后的"兼容"未定义**：若代码或数据管道在中断期间变更，当前只有 format_version 一道闸，语义兼容性（模型结构变化）依赖 torch.load 自然报错。

## 6. 文件清单

- `toy/recovery-baseline.py` —— 原始基线（未改动；写入 cwd 的 checkpoint.json）
- `toy/check_riskiest_assumption.py` —— 风险假设最小检查
- `toy/train_recoverable.py` —— 主实验（AI 辅助实现，负责人验收）
- `mlp/train_mlp.py` —— T5 真实模型层（AI 辅助实现，负责人验收；解释器：集群共享环境 pytorch_gpu_2.0.1）
- `toy/results/equivalence.md`、`toy/results/negative_tests.txt`、`toy/results/run_log.md` —— 机制层运行记录
- `mlp/results/`（equivalence.md、negative_tests.txt、run_log.md、checkpoints/、logs/）—— 真实模型层运行记录
- `mlp/data/` —— MNIST 数据（ossci S3 镜像下载，不入库）
- `docs/PLAN.md` —— 共同规划与验收记录；`docs/DECISIONS.md` —— 四问决策回答（已整合原过程版 DECISION-JOURNAL.md）
