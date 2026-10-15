# 共同规划 — 可恢复训练流水线实验

> 本文档是跨工作会话维护的共同规划与独立记录。每次会话结束更新「会话记录」「进行中的 AI 作业」「后续行动」「验收记录」。
> 任务说明见 `brief/recovery-brief.md`（中文版 `brief/recovery-brief-zh.md`）。

- 最后更新：2026-10-16（提交前全面校对，会话 1–4）
- 当前阶段：全部计划事项完成（P0/P1/P2 清零）；交付物为 `docs/PLAN.md`、`docs/RISK-ASSUMPTION.md`、`docs/DECISIONS.md` 三份文档（2026-10-16 起）

## 1. 目标与范围

在任务书范围内（小型 checkpoint/resume 实验，CPU 即可）：

1. 保存模型、优化器、训练步数和相关随机状态；
2. 对比中断/恢复与不间断运行，报告实际等价性检查；
3. 对缺失或不兼容的 checkpoint 显式处理；
4. 形成可捍卫的决策（显式失败 / 部分恢复 / 从头重启）+ 风险披露，支撑决策日志四个问题。

**不在范围内**：真实大模型/GPU 训练、分布式训练恢复、生产级训练框架。

## 2. 风险最高的假设（优先核验）

| 假设 | 最小检查 | 结果 | 结论与理由 |
|---|---|---|---|
| A1：完整保存并恢复全部状态后，中断/恢复运行能与不间断运行在浮点位级等价 | 已做：基线 toy 上分段（4+6）训练 + JSON 往返 vs 不间断 10 步 | **通过（位级一致）**：两者 step=10、weight=2.6778774528，`==` 与 `repr` 均相等 | 在确定性单线程 CPU toy 上成立。理由：更新是确定浮点运算、JSON 序列化用 repr 可无损往返 float。但**不能外推**：基线只存 `step`+`weight`（无优化器、无 RNG），toy 无随机性，故未建立真实训练语义下的等价保证 |
| A2：基线 checkpoint 足以支撑「忠实恢复」 | 检查 `checkpoint.json` 内容 | **不成立**：仅含 `{'step','weight'}`，缺优化器与随机状态 | 基线只能支持 toy 级恢复，正式实现必须扩展状态保存（见 T2） |
| A3：缺失/不兼容 checkpoint 可被静默容忍 | T3/selftest 负例（机制层 5 例 + 真实模型层 5 例） | **不成立**（即期望结果）：全部被显式拒绝，exit=2、stderr 可读、无 Traceback | 显式失败语义在两层一致落地，无静默继续 |

本节为摘要；独立交付文档见 `docs/RISK-ASSUMPTION.md`（假设识别理由、最小检查设计、实测输出、结论理由与后续核验链）。

检查脚本：`toy/check_riskiest_assumption.py`；复现：`cd recovery-baseline && python3 toy/check_riskiest_assumption.py`。

## 3. 工作分解（优先级 / 依赖 / 状态）

优先级：P0 = 提交必需；P1 = 重要；P2 = 可选。

| ID | 类别 | 任务 | 优先级 | 依赖 | 状态 |
|---|---|---|---|---|---|
| T0 | 评估 | 运行基线并核验输出 | P0 | 无 | ✅ 完成（`{'step': 10, 'weight': 2.6778774528}`） |
| T1 | 评估 | 风险最高假设最小检查（A1–A3） | P0 | T0 | ✅ 完成（A1 通过、A2 证伪，记录见第 2 节） |
| T2 | 实现 | 扩展 checkpoint：模型 + 优化器 + 步数 + RNG 状态；实现 resume 与版本/兼容性校验 | P0 | T0 | ✅ 完成（会话 A） |
| T3 | 实现 | 缺失 / 不兼容 checkpoint 的显式处理 + 负例测试 | P0 | T2 | ✅ 完成（会话 A；后经 O2 扩至 5/5，含倒退 resume 与字段类型校验） |
| T4 | 实现 | 等价性对比：中断/恢复 vs 不间断，输出对比表（含容差与位级两档） | P0 | T2 | ✅ 完成（会话 A，双切分点逐步位级一致） |
| R1 | 阅读 | 调研确定性训练的可行保证边界（如 cuDNN/多线程非确定性、Python RNG 恢复语义） | P1 | 无 | ✅ 完成（会话 B，6 个权威来源，含可引用的保证边界声明） |
| R2 | 阅读 | 整理真实框架（PyTorch checkpoint 语义）作对照，明确本文实验的诚实边界 | P2 | R1 | ✅ 完成（并入 docs/REPORT.md「已验证 vs 未建立」一节，不单独立项） |
| E1 | 评估 | 汇总结果表、失败案例、已知局限，区分「已验证」与「未建立」的保证 | P0 | T3, T4 | ✅ 完成（`docs/REPORT.md`） |
| C1 | 沟通 | 撰写决策日志四问（优先级/利益方权衡/不确定性/担当反思） | P0 | E1 | ✅ 完成（`docs/DECISIONS.md`；原过程版 DECISION-JOURNAL.md 已于 2026-10-16 并入其中） |
| REC | 评估 | 环境侦察：torch/torchvision 可用性、外网连通性（PyPI / torch wheel 源 / MNIST 镜像源） | P0 | 无 | ✅ 完成（见 §7 会话 4） |
| C2 | 沟通 | 提交前自检：可复现（干净环境重跑全部命令）、AI 辅助披露 | P0 | 全部 | ✅ 完成（2026-10-15 全命令序列重跑，输出与记录一致） |
| T5 | 实现 | 真实模型层：PyTorch MLP + MNIST，确定性开关全开；checkpoint 存 model/optimizer/scheduler state_dict、epoch、torch/numpy/python 三处 RNG、DataLoader generator；三档恢复对比（全状态 / 丢优化器 / 丢 RNG），报告 allclose 容差级 + 最终指标差；负例语义复用；输出至 `mlp/results/` | P1 | REC 完成；环境定案 | ✅ 完成（会话 C，负责人独立复跑：compare 重跑输出 md5 不变，selftest 5/5） |

## 4. 进行中的 AI 作业

| 时间 | 作业 | 工具/范围 | 状态 |
|---|---|---|---|
| 2026-10-15 | 中文版任务书翻译 | 助手直写 | ✅ 完成（`brief/recovery-brief-zh.md`） |
| 2026-10-15 | 风险假设检查脚本与共同规划初稿 | 助手直写 | ✅ 完成 |
| 2026-10-15 | 会话 A：T2–T4 实现 | coder 子 agent | ✅ 完成（`toy/train_recoverable.py` + toy/results/；负责人已独立复跑验证） |
| 2026-10-15 | 会话 B：R1 阅读调研 | explore 子 agent（只读） | ✅ 完成（结论并入 docs/REPORT.md） |
| 2026-10-15 | E1 汇总与 C1 决策日志 | 负责人亲笔（判断性工作，不委派） | ✅ 完成 |
| 2026-10-15 | 环境侦察（T5 前置） | 负责人直查（import 检查 + curl 连通性） | ✅ 完成：torch/torchvision 未装；PyPI 与 download.pytorch.org 通（200）；ossci-datasets S3 可达（403=桶列表正常响应，文件 GET 待实测）；yann.lecun.com 超时（弃用） |
| 2026-10-15 | 会话 C：T5 真实模型层实现 | coder 子 agent | ✅ 完成（`mlp/train_mlp.py` + mlp/results/；MNIST 经 ossci 镜像下载成功；compare 输出重跑字节级一致；5/5 负例） |
| 2026-10-16 | 四问独立定稿（docs/DECISIONS.md） | 负责人亲笔（判断性工作，不委派） | ✅ 完成（按评估方四问完整提示结构：立场与牺牲 / 两选项权衡与反对意见 / 事实假设区分与承诺变更条件 / 一项 AI 建议的修改与改进） |

（后续会话在此追加；每条应可独立审计——写明输入、产出文件与验证方式。）

## 5. 后续行动

1. [P1] T5 真实模型层（会话 C）：✅ 已完成（输出至 `mlp/results/`，保证分层预写执行到位）。
2. [P1] T5 验收后更新：✅ 已完成（PLAN 验收记录 + AI 作业表 + 会话记录；决策日志第 3 问 T5 补充已写入，现并入 `docs/DECISIONS.md`；负责人独立复跑后签字）。
3. [P1] 数据集源决策：✅ 已解决——MNIST 经 torchvision 默认 ossci S3 镜像下载成功，未回退合成数据。
4. 早前行动（已完成）：C2 ✅、切分点 1–9 扫掠 ✅、R1 对照表 ✅（会话 2–3）。

## 6. 验收记录与结果

| 日期 | 验收项 | 命令 | 结果 |
|---|---|---|---|
| 2026-10-15 | 基线可运行且输出符合预期 | `python3 toy/recovery-baseline.py` | 通过：`{'step': 10, 'weight': 2.6778774528}` |
| 2026-10-15 | 风险假设 A1：分段恢复位级等价 | `python3 toy/check_riskiest_assumption.py` | 通过：uninterrupted 与 resumed 位级相等 |
| 2026-10-15 | 风险假设 A2：基线 checkpoint 内容 | 同上脚本输出 checkpoint keys | 证伪假设：仅存 `step`+`weight`，缺优化器/RNG → 触发 T2 |
| 2026-10-15 | T2：全状态 checkpoint + resume | `python3 toy/train_recoverable.py run --total 10 --save-checkpoint …` 后 `--checkpoint …` 续跑 | 通过：手动中断/恢复后 weight/velocity 与不间断运行完全一致 |
| 2026-10-15 | T4：双切分点逐步等价对比（含 RNG 覆盖测试） | `python3 toy/train_recoverable.py compare --total 10 --splits 4 7` | 通过：split 4 与 split 7 均 BIT-EXACT=True（10/10 步）；部分恢复最大偏差 0.0240948 / 0.00893984 |
| 2026-10-15 | T3：四类负例显式失败 | `python3 toy/train_recoverable.py selftest` | 通过：missing file / wrong version / corrupt JSON / missing key 均 exit=2 且 stderr 可读；selftest exit=0 |
| 2026-10-15 | 负责人独立复验 | 亲自重跑 compare + selftest，抽查源码与 negative_tests.txt | 通过：与子 agent 报告一致，无自证水分（RNG 覆盖设计真实有效） |
| 2026-10-15 | R1：等价性保证边界调研 | 会话 B 只读调研 | 通过：PyTorch 官方文档等 6 个来源，产出可引用边界声明 |
| 2026-10-15 | C2：提交前干净环境自检 | 按 docs/REPORT.md 第 1 节命令序列逐一重跑 | 通过：基线/风险检查/手动中断恢复/compare/selftest 全部输出与记录一致；AI 辅助披露已写入 docs/REPORT.md 与决策日志（现 `docs/DECISIONS.md`） |
| 2026-10-15 | O2：修复显式失败缺口（倒退式 resume 静默成功、字段类型错误抛 traceback） | 恢复会话 A 修复；selftest 5/5 + 两个新错误模式实测 | 通过：5 例均 exit=2、stderr 可读且无 Traceback；回归一致（BIT-EXACT 与偏差值不变） |
| 2026-10-15 | O1：切分点 1–9 全扫掠 | `compare --total 10 --splits 1..9` | 通过：9/9 BIT-EXACT=True；部分恢复在全部 9 个切分点可测发散（0.00095–0.0241） |
| 2026-10-15 | O3：toy vs PyTorch 生产保证对照表 | 写入 docs/REPORT.md §4.1 | 通过：6 维度对照，来源为会话 B 调研（PyTorch 官方文档等） |
| 2026-10-15 | REC：T5 环境侦察 | `import torch`/`import torchvision`；curl PyPI / download.pytorch.org / ossci S3 / yann.lecun.com | 通过：torch 与 torchvision 未安装；PyPI(200) 与 download.pytorch.org(200) 可达 → pip 装 CPU 版可行；ossci S3 可达（403 为桶列表正常响应，文件级 GET 待 T5 实测）；yann.lecun.com 超时弃用 |
| 2026-10-15 | T5：三档恢复对比 | `mlp/train_mlp.py compare --epochs 3 --split-epoch 1 --split-batch 47`（集群 pytorch_gpu_2.0.1 解释器；会话 C 运行；负责人后台独立复跑） | 通过：全状态恢复两个切分点 max param diff=0.0、allclose=True（观察到位级一致，按预写边界仅作观察）；丢优化器 Δacc −1.2%/−2.0%；丢 RNG Δacc +0.8%/+1.4%；复跑后 mlp/results/equivalence.md md5 不变（3217ce1a…） |
| 2026-10-15 | T5：负例 5/5 + 代码抽查 | `mlp/train_mlp.py selftest`（集群 pytorch_gpu_2.0.1 解释器）；抽查 RNG 存/载、clobber、shuffle generator rewind 逻辑 | 通过：5 例 exit=2 可读无 Traceback；恢复前三处 RNG 被覆盖以证明还原真实；预分段一致性是验证而非假设 |

## 7. 会话记录

**会话 1（2026-10-15）**：建立目录内工作副本；运行基线；完成风险最高假设（A1）最小检查并通过；发现基线 checkpoint 不完整（A2），据此排定 T2 为最高优先实现项；建立本共同规划。

**会话 2（2026-10-15）**：派发会话 A（coder，T2–T4 实现）与会话 B（explore，R1 调研）两个子 agent 任务会话。负责人独立复跑全部命令验收：全状态恢复在切分点 4、7 逐步位级一致；部分恢复（丢 RNG）实测发散；四类 checkpoint 负例全部显式报错（exit=2）。抽查源码确认 RNG 存/载无隐藏消耗、无自证设计。随后负责人亲笔完成 E1（`docs/REPORT.md`）与 C1（当时的 `DECISION-JOURNAL.md`，已并入 `docs/DECISIONS.md`）。

**会话 3（2026-10-15）**：负责人代码审查发现两个诚实缺口——(a) resume 目标步数小于 checkpoint step 时静默"成功"；(b) 字段类型错误抛裸 traceback 而非可读报错。恢复会话 A 修复（训练语义与 RNG 流未变），selftest 增至 5 例且全部断言无 Traceback；负责人回归复跑通过。随后完成 O1（切分点 1–9 全扫掠：9/9 位级一致，部分恢复 9 点全部可测发散）与 O3（toy vs PyTorch 对照表写入 docs/REPORT.md §4.1）。P0、P2 事项全部清零。

**会话 2 收尾**：C2 提交前干净环境自检通过——docs/REPORT.md 第 1 节全部 7 条命令逐一重跑，输出与记录一致；AI 辅助披露已写入 docs/REPORT.md 与决策日志（现 `docs/DECISIONS.md`）。至此 P0 事项（T0–T4、E1、C1、C2）全部完成，仅剩 P2 可选项（更多切分点、对照表）。

**会话 4（2026-10-15）**：新增 T5 真实模型层（P1）。环境侦察（便宜优先）：Python 3.13.9、numpy 2.5.3 在位；torch/torchvision 未安装；网络部分可达——PyPI 与 download.pytorch.org 返回 200（pip 安装 torch CPU 版可行），ossci-datasets S3 返回 403（桶列表的正常响应，文件级 GET 待 T5 实测），yann.lecun.com 超时（弃用该源）。决策：按用户指令走 hpc-environment 技能建环境装 torch；数据集优先 MNIST（torchvision 默认 S3 镜像），若下载不通则暂停报告选项，不静默退回合成数据。玩具层（toy/train_recoverable.py 与 toy/results/）保持不动，仍作为机制层位级证据。

**环境定案（hpc-environment 技能，会话 4）**：集群共享 anaconda3 已有 `pytorch_gpu_2.0.1` 环境（python 3.11.4、torch 2.0.1+cu118、torchvision 0.15.2、numpy 1.24.1），登录节点 CPU 路径实测可用（cuda.is_available()=False），**直接复用、零安装**（技能优先复用原则）。pip 全局镜像为阿里云（安装后备方案，本次未用）。T5 一律使用该环境的 python 解释器（本机绝对路径略；即集群 anaconda3 的 envs/pytorch_gpu_2.0.1）。

**T5 实施与验收（会话 4 续，2026-10-15）**：派发会话 C（coder）实现真实模型层。关键结果：全状态恢复在两个切分点（epoch 1 边界 / 全局 batch 47 中周期点）max param diff=0.0、allclose=True，并观察到位级一致（按预写边界仅作观察，不升级为保证）；两类部分恢复均改变最终测试精度（|Δacc| 0.8–2.0%）。开发中发现并修复两个真实 bug（中周期停止的 epoch/step 不一致；torch 2.0.1 RandomSampler 每轮重新 randperm 导致的中周期续跑错位——通过快照并回绕 shuffle generator 修复），已记入 run_log。负责人独立复跑 compare：exit=0 且 equivalence.md md5 与原厂一致；selftest 5/5 亲验；抽查确认 RNG 覆盖、四源 RNG 存载、预分段位级验证（验证而非假设）均真实。

**会话 5（2026-10-16）**：提交前全面校对。交付物收敛为三份独立文档：`docs/PLAN.md`（工作规划）、`docs/RISK-ASSUMPTION.md`（风险最高的假设）、`docs/DECISIONS.md`（四问个人回答定稿，按评估方完整提示结构亲笔重写）。修正 PLAN 过时状态 4 处（头部阶段、A3 行、T3 计数、后续行动勾选）。四问文档要点：优先级「可信>省时>省力」并以 O2 修复等具体决策佐证；相关方权衡重点比较「显式失败 vs 打标部分恢复」，回应沉没成本与"偏差很小"两条反对意见；不确定性下承诺分层保证并预写降级触发条件；反思聚焦"否决 AI 初版对比逻辑（漏检恢复点）"一事，收获是诚实性应写成可执行验收断言，并说明个人承担签字风险。当日随后合并 `DECISION-JOURNAL.md`（过程版）入 `docs/DECISIONS.md`：T5 签字措辞与"接受/拒绝"两项亲自决策并入，前者删除，PLAN 与 REPORT 中的全部引用已更新。

**会话 6（2026-10-16）**：目录结构整理并推送 GitHub（Evan7101/recoverable-training-pipeline）。新结构：`docs/`（四份文档）、`brief/`（任务书）、`toy/`（机制层脚本+results）、`mlp/`（train_mlp.py+results+data）；脚本输出路径改为相对脚本自身定位，任意 cwd 下命令可用；`git mv` 保留历史。整理后全量复跑验证：toy 三条命令 + mlp compare/selftest 全部通过，`mlp/results/equivalence.md` md5 不变（3217ce1a…），toy 全扫掠恢复后与整理前逐字节一致；两份 negative_tests.txt 仅临时目录名随重跑变化（固有行为）。已提交（见 git log），负责人独立抽验后签字。
