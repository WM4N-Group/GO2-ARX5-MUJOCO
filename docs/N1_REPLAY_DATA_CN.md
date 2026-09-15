# N1 技能快照、物理重放与候选 Pilot

范围说明（2026-09-15）：本文是N1旧生产技能快照与pilot的阶段记录，功能随后发布于f857437并由008a795代码基线继承。当前项目优先级、机器和新模型以 [主交接](AGENT_HANDOFF_CN.md) 为准；新移动箱组合尚未接入这里的生产记录/快照流程，不能把旧回放成绩外推给新Box runtime。历史数据须使用其对应代码指纹，不能修改manifest绕过兼容检查；大规模扩容仍暂缓。

验证日期：2026-09-11。N1 实现基于 `6c71d1a`，代码已同步并在 yuanyue 服务器验证，发布版本以 Git 历史为准。本轮没有重新训练 actor；快照和 pilot 数据保留在服务器，不纳入 Git。

2026-09-12 补充：控制步过程事件和失败原因已实现，schema v2 分别在本地与新服务器通过 32 项合约及物理数据验证，详见 [技能过程事件](N1_SKILL_EVENTS_CN.md)。下文 600 请求的服务器 pilot 是历史 v1 数据；新版服务器另有 60 请求验收，不能将两次结果混写。

## 1. 已完成范围

- 在 executor 接受动作后的技能开始边界，通过可选 `on_skill_start` 保存物理和控制器状态；原有默认执行接口保持兼容。
- `SimulatorSnapshot.capture()` 创建独立的 MuJoCo model/data、runtime、策略和环境对象；保留控制历史、CLIMB 延迟队列、实例延迟参数和随机数状态。
- `save()` / `load(trusted=True)` 将快照持久化；`rollout()` 在独立分支中执行一个 NAV/PUSH/CLIMB，复用现有技能和低层控制器，不移动主世界对象、不增加主 episode 的 reset。
- 复杂课程的 `--snapshot-dir` 为 JSONL 中的每个技能提供起点快照引用和 SHA-256。
- 批量采集器从起点生成原动作、邻近参数、边界参数及少量非法请求，保存真实物理结果。它是数据工具，尚未替代生产规则 Oracle 或实现几何候选搜索。

实现入口：[snapshot.py](../mujoco/reconfigurable_navigation/data/snapshot.py)、[snapshot_io.py](../mujoco/reconfigurable_navigation/data/snapshot_io.py)、[候选定义](../mujoco/reconfigurable_navigation/data/candidates.py)、[重放命令](../mujoco/replay_skill_records.py)、[采集命令](../mujoco/collect_skill_candidates.py)。

## 2. 快照契约

### 保存的状态

| 层 | 内容 |
| --- | --- |
| MuJoCo | 完整 model、MjData integration state 和原生计算缓存，包括 warm-start、接触及传感器相关状态 |
| NAV runtime | 三帧历史、last action、速度/末端命令、控制参数、关节映射及独立策略 |
| CLIMB runtime | 延迟 target history、last action、速度命令、延迟步数、速度限制及独立策略 |
| 环境/规划配置 | 环境子类字段、Capability、reset_count、环境随机数生成器、已有规划和安全配置 |
| 随机数 | Python、全局 NumPy、Torch CPU 状态；`rollout()` 使用临时隔离作用域并在结束后还原调用方状态 |

恢复不额外调用 `mj_resetData()` 或 `mj_forward()`。现场观测可能依赖最后一次物理步留下的缓存，额外重算会改变首帧观测；使用原生 `mj_copyData` 和原生状态字节保留这些内容。

### 明确限制

- 支持的是当前 GO2 场景和 CPU TorchScript actor 的控制/技能边界，不是任意 Python 对象、GPU actor、MuJoCo plugin 或全局 MuJoCo callback。
- 完整持久化回放面向“技能开始前”。不会保存 `run()` 的局部调用栈、正在执行的 Skill 对象、当前技能内步数或完整 episode 的累计执行记录/预算。不能把 `fork().run()` 当作原 episode 的无缝继续，它仍会执行原来的启动逻辑。
- `rollout(action)` 重建并执行一个技能；正常完成后可以使用 `next_snapshot` 接下一个动作。预算截断没有可恢复的中途 Skill 状态，所以不返回后继快照。
- `fork()` 是独立物理对象复制的低层接口；需要全局随机数隔离的候选应使用 `rollout()`。同进程的多线程并发不在支持范围，批量采集采用独立进程。
- 归档保存 NumPy 数组时使用 `allow_pickle=False`，对象属性采用受限类型 JSON 编码；MuJoCo 原生状态和 TorchScript 本身仍需信任。只加载自己生成的文件，命令要求显式 `--trusted`。校验和用于发现损坏，不代表第三方文件可信。
- 加载检查 schema、Python/MuJoCo/PyTorch/NumPy 版本、架构、字节序、Torch 线程数、相关运行代码指纹和成员 SHA-256。已有归档不保证能在修改后的 runtime 或其他版本上加载，不应通过改 manifest 绕过检查。
- 每个归档包含模型和 actor，50 个当前快照约 3.4 GB；这是正确性优先的首版，尚未做共享资产去重。快照写入不覆盖已存在文件。

## 3. 运行方式

以下命令在新服务器执行，Python 环境已经安装。所有命令用单行，避免粘贴续行符后的不可见字符。

```bash
cd /mnt/yuanyue/GO2-ARX5-MUJOCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GO2_PYTHON=/mnt/yuanyue/envs/go2-mujoco/bin/python
```

检查接口：

```bash
"$GO2_PYTHON" mujoco/check_skill_transitions.py -v
"$GO2_PYTHON" mujoco/check_replay_snapshots.py -v
"$GO2_PYTHON" mujoco/check_skill_candidates.py -v
```

创建一个新数据目录，采集 10 个任务的起点快照：

```bash
export N1_OUTPUT="$(mktemp -d /mnt/yuanyue/data/n1-run-XXXXXX)"
"$GO2_PYTHON" mujoco/check_complex_course.py --seeds 10 --record-jsonl "$N1_OUTPUT/transitions.jsonl" --snapshot-dir "$N1_OUTPUT/snapshots"
```

在新的进程中重放首个 episode 的五个技能，各三次：

```bash
"$GO2_PYTHON" mujoco/replay_skill_records.py --record-jsonl "$N1_OUTPUT/transitions.jsonl" --indices 0 1 2 3 4 --repeats 3 --trusted
```

`--indices` 是 JSONL 的零基下标；记录比较包括动作、前后观测、技能状态、前序技能和仿真时间，默认绝对容差为 `1e-6`。

从每个起点采集 12 个候选，使用 4 个独立 CPU worker：

```bash
"$GO2_PYTHON" mujoco/collect_skill_candidates.py --record-jsonl "$N1_OUTPUT/transitions.jsonl" --output-dir "$N1_OUTPUT/candidates-pilot" --candidates-per-snapshot 12 --workers 4 --seed 0 --max-control-steps 5000 --trusted
```

输出目录必须不存在，防止覆盖其他数据。每个 worker 限制为一个 Torch/BLAS 线程；不要将线程数不匹配的旧归档强行加载。

## 4. 数据布局与标签

```text
N1_OUTPUT/
  transitions.jsonl
  snapshots/<episode-id>-<skill-index>.snapshot
  candidates-pilot/
    manifest.json
    candidates.jsonl
```

采集未完成时保留 `complete=false` 的 manifest 和 `candidates.partial.jsonl`，不能当成完整训练集；成功后才原子切换到正式结果名。当前未实现断点续采，重试应使用新的输出目录。

每条候选保存来源 record index、episode ID、场景种子、候选种子、来源类型、动作、预算、快照引用/hash、runtime 版本及代码指纹。manifest 保存来源 JSONL 的 hash、候选生成器/采集代码 hash、结果 hash、数量及耗时。移动数据时需连同源 JSONL 与快照目录保留相对路径。

| 结果 | executed | transition | skill_success | 标签解释 |
| --- | --- | --- | --- | --- |
| succeeded | true | 实际观测转移 | true | 已执行技能成功 |
| failed | true | 实际观测转移 | false | 已执行技能失败，详细失败归因仍待补全 |
| rejected | false | null | null | 前置/状态条件拒绝，不伪造零位移动力学样本 |
| truncated | true | 截断时观测 | null | 达到采集预算，不能按完整技能失败训练 |

截断的位移仅表示“有限时间内的部分转移”，不是技能结束状态；使用动力学样本时必须同时考虑时间和截断 mask。当前 reachability 与 support_stable 的有效标记均为 false，没有从单次技能结果捏造后续可达或稳定支撑标签。

非法请求用于验证拒绝行为，与实际物理失败分别统计。候选类型中的 boundary_parameter 只是较大参数扰动，不代表已经完成能力边界标定或几何可行性过滤。

## 5. 本次实测

服务器产物根目录：

```text
/mnt/yuanyue/data/n1-replay-btso9L/
```

- 10 个源任务全部成功，50 个起点快照和源 transition；首个任务五个阶段均在独立进程中重放三次通过。
- 快照合约 10/10、技能边界记录 7/7、候选合约 6/6，共 23 项通过。
- 连续 100 控制步测试中动作一致，状态绝对误差不超过 `1e-10`；真实技能记录使用 `1e-6` 比较。未宣称任意机器/版本上的逐位一致。
- 完整候选采集为 600 个请求，4 个 worker 耗时约 242.5 秒；结果 JSONL 约 3.98 MB，不含共享快照。
- 全部 50 个 reference 候选与其原始记录逐字段比较通过，其中源技能本身为 48 次成功和 2 次失败。
- 审查通过来源/输出 hash、候选唯一 ID、每起点 12 个候选、标签 mask 和控制时间预算。
- 默认未开启记录/快照的回归仍通过：CLIMB 切换 10/10、PUSH 5/5。

| 技能 | 请求数 | 成功 | 执行失败 | 前置拒绝 |
| --- | ---: | ---: | ---: | ---: |
| NAV | 360 | 280 | 50 | 30 |
| PUSH | 120 | 79 | 31 | 10 |
| CLIMB | 120 | 82 | 20 | 18 |
| 总计 | 600 | 441 | 101 | 58 |

因此实际执行记录为 **542 条**，拒绝的 58 个请求不计入动力学样本数量。250 个邻近参数候选产生 237 次成功、13 次失败；250 个边界参数候选产生 156 次成功、86 次失败和 8 次前置拒绝；50 个明确非法请求全部被拒绝。

额外 `budget-check/` 在一步控制预算下得到 3 条真实截断记录和 1 条拒绝，没有任何物理失败标签。主 pilot 中没有预算截断；不要为了凑比例改标签。

## 6. 下一步与未完成项

pilot 数量已达到最初建议的 500-1000 条执行样本区间，但它只有一个固定布局场景族、10 个源 episode、50 个起点。600 个相关候选不是 600 个独立场景，也不构成 held-out 评估。manifest 明确记为 `pilot_unsplit`，同一起点及其能力/参数变体不能跨训练和测试集。

接下来按以下顺序补完 N1/N2：

1. 控制步接触/状态事件及无进展/超时等原因已在本地和服务器验证；继续完善逐物理步覆盖、失稳与支撑判据。保留 unknown/有效标记，不单看结束时姿态推断全过程安全。
2. 参数化场景与物理一致的真值对象，建立能力 profile 和多场景族划分，区分直接可达、需重构、已证明不可行。
3. 从参数扰动转为几何 grounding 的推面、退出区、支撑区域候选，并在相同预算下做物理候选比较。
4. 根据覆盖和学习曲线决定数据扩容，之后才训练特权状态世界模型。多步价值、reachability 和 support_stable 仍需单独定义与验证。

当前生产 planner、actor 和复杂课程的物理参数未因候选采集改变；并未接入世界模型、VLM、RGB-D 或 JUMP。