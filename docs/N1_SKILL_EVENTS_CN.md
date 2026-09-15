# N1 技能过程事件与失败原因

范围说明（2026-09-15）：本文记录随后发布于9ac5b07的旧生产技能事件schema v2，当前008a795已包含该实现；下述实验不在本次文档整理中重跑。新Box组合尚未接入此executor事件/快照链，不能将它的独立JSON报告当成已有N1训练数据。当前任务与版本见 [主交接](AGENT_HANDOFF_CN.md)，数据扩容与世界模型训练仍暂缓。

验证日期：2026-09-12。本轮实现基于 `f857437`，已同步并分别在本地和 yuanyue 服务器隔离 MuJoCo 环境完成验证，发布版本以 Git 历史为准。SSH 公钥登录已通过全新连接验证，后续不再依赖临时密码会话。此前服务器的 600 请求 pilot 仍是旧版数据，见 [快照与数据记录](N1_REPLAY_DATA_CN.md)。

## 1. 实现范围

`--record-jsonl` 现在输出 schema v2：保留前后观测、动作、耗时和中断字段，增加过程事件、过程标签与结束原因。实时 executor 和独立快照 rollout 使用同一个事件采集器。默认未启用记录的执行不创建采集器。

实现入口：

- [events.py](../mujoco/reconfigurable_navigation/data/events.py)：接触/状态事件边沿、PUSH 阶段记录。
- [transition.py](../mujoco/reconfigurable_navigation/data/transition.py)：schema v2 与结束原因序列化。
- [executor.py](../mujoco/reconfigurable_navigation/runtime/executor.py)：技能起点及每个控制步后的采样。
- [candidates.py](../mujoco/reconfigurable_navigation/data/candidates.py)：候选结果及训练标签掩码。
- [check_skill_events.py](../mujoco/check_skill_events.py)：真实技能状态机的失败归因测试。

现有技能阈值、控制命令、actor、物理参数和终止判定保持原样。这一轮增加记录能力，不为 NAV 新增碰撞停止条件，也不重新训练策略。

## 2. 事件和采样语义

| 字段 | 定义 |
| --- | --- |
| `events` | 事件按时间排列；接触或异常出现时 `active=true`，消失时 `active=false` |
| `kind` | `contact`、`end_effector_contact`、`body_contact`、`illegal_collision`、`invalid_robot_state`，以及 PUSH 的 `phase` |
| `sim_time` / `control_step` | MuJoCo 仿真时间与从技能起点计数的控制步；起点是第 0 步 |
| `stage` | `preparation` 或 `execution`，区分 CLIMB 准备与实际技能执行 |
| `object_id` | 接触对象 ID；不适用时为 null |
| `skill_phase` | PUSH 的 align/contact/push/verify/retreat；其他技能为 null |
| `event_sample_count` | 包含技能起点及每次控制推进后的观测；CLIMB 准备步也计入 |
| `process_labels` | 每类接触/异常是否曾在采样中出现；没有采样时为 null |
| `event_sampling` | 有采样时固定为 `control_boundary` |
| `process_label_scope` | 完整执行为 `executed_skill`；中断/截断为 `observed_prefix` |

事件只在边沿变化时写入，持续接触不会每步重复占据一条事件。即使结束时碰撞已消失，过程中出现过的碰撞标签仍为 true。PUSH 的对象接触、指尖接触和机身/腿部接触分别记录；允许的脚底支撑不作为非法碰撞。

当前采样粒度是控制步，通常为 0.02 秒，无法保证捕获两个控制边界之间的短暂物理接触。`process_labels=false` 仅表示这些采样中未观察到事件，不能证明连续物理轨迹完全无接触。候选标签掩码针对这一采样定义有效。

`invalid_robot_state` 沿用环境现有的有限数值和 base 高度判定，不等同于完整的侧翻、倾斜角或支撑稳定性检测。没有从最终姿态推断全过程安全，也没有将该字段当作 `support_stable` 标签。

## 3. 结束原因和标签掩码

原因来自技能实际失败分支，`reset()` 清除上次原因。成功为 `target_reached`；没有具体原因的失败为 `unknown_failure`，其监督用 `failure_reason` 为 null。

| 原因 | 适用条件 |
| --- | --- |
| `invalid_robot_state` | NAV/PUSH/CLIMB 观察到无效状态 |
| `illegal_collision` | PUSH/CLIMB 原有碰撞失败分支 |
| `timeout` | 技能自身总步数上限 |
| `contact_timeout` | PUSH 接触阶段未及时建立指尖接触 |
| `no_progress` | PUSH 推动阶段累计无进展超出已有阈值 |
| `target_not_maintained` | PUSH 验证阶段目标位置未保持 |
| `object_missing` / `object_not_movable` / `object_too_heavy` | PUSH 执行时的目标对象或能力检查失败 |
| `body_contact` / `degenerate_push_target` | PUSH 的机身/腿接触或退化目标失败 |
| `executor_timeout` | 技能超过 executor 外层循环上限仍未结束 |
| `execution_interrupted` | 实时执行的外部回调中断 |
| `rollout_budget_exhausted` | 独立候选 rollout 的控制步预算耗尽 |

中断和预算截断的 `skill_success`、`failure_reason` 均为 null。候选前置条件拒绝没有 transition，也没有过程标签。截断样本已经观察到的正事件可以用于“曾发生”标签；未观察到的事件对应掩码为 false，避免将未完成技能当作完整负样本。未知的 reachability 和 support_stable 掩码继续为 false。

原有 `SkillExecutionRecord` 和 `SkillReplay.reason` 接口保持兼容；候选 JSON 的 `reason` 使用具体结束原因。实时中断记录的重放通过控制步预算模拟停止位置，比较时排除外部中断触发原因的差异，不声称恢复了原 GUI 回调。

## 4. 版本与运行

transition、候选 JSON 和候选 manifest 均使用 schema v2。快照容器本身仍是 v1，运行代码指纹新增覆盖事件模块。旧归档不会绕过版本/代码指纹检查；重放旧 pilot 应使用产生它的代码和环境，验证新版应重新采集起点快照。不要直接把旧数据改成 v2 或把未知过程标签补成 false。

在激活的 MuJoCo 环境、仓库根目录运行：

```bash
python -m unittest discover -s mujoco -p 'check_skill_*.py' -v
python mujoco/check_replay_snapshots.py -v
```

采集、重放和候选命令沿用 [N1 使用方式](N1_REPLAY_DATA_CN.md#3-运行方式)，但必须使用新的输出目录。单个归档仍要求信任自生成的原生状态和 TorchScript。

## 5. 本地验证结果

本次使用本地 Python 3.11 / MuJoCo 3.12 / CPU PyTorch 隔离环境，不能把以下结果计作 yuanyue 服务器新增验收。

- 22 项事件、边界记录和候选合约测试通过；10 项真实快照合约测试通过，共 32 项。
- 未开启记录的回归：复杂课程 10/10、CLIMB 切换 10/10、物理 PUSH 5/5。
- 带记录源任务 1/1 完成，产生 5 个快照、49 条过程事件。5 条技能记录为 4 次成功和 1 次超时失败，任务整体仍成功，标签未被任务结果覆盖。
- 五个技能分别在新进程中重放三次，状态、时间、事件和原因逐字段比较通过，绝对容差 `1e-6`。
- 60 个物理候选请求：42 次成功、11 次失败、7 次前置拒绝，共 53 条执行记录；失败原因是 4 次 `no_progress` 和 7 次 `timeout`。
- 五个 reference 候选完整重现源记录；事件时间、控制步数、标签、唯一 ID 和结果文件 hash 审查通过。源 PUSH 的五个阶段均被记录。
- 一步预算验收：6 条真实截断、3 条前置拒绝；CLIMB 在准备阶段截断时 `skill_steps=0`，仍记录一次控制推进和两次观测，没有物理失败标签。

产物位于仓库外：

```text
/home/yuanyue/re-nav/artifacts/n1-events-FfYeP0/
  transitions.jsonl
  snapshots/
  candidates/candidates.jsonl
  candidates/manifest.json
  budget-check/candidates.jsonl
  budget-check/manifest.json
```

这组样本用于验证标签链路，仍只有一个固定布局、一个源任务，不能与旧服务器 pilot 混为独立场景，也不构成泛化评估。

## 6. 服务器验证结果

2026-09-12，使用新服务器 `/mnt/yuanyue/envs/go2-mujoco` 完成以下独立验收：

- 22 项事件/标签合约和 10 项真实快照合约通过，共 32 项。
- 默认复杂课程 10/10、CLIMB 切换 10/10、物理 PUSH 5/5。
- 新采集的源任务 1/1 完成，五次技能均成功，产生五个快照和 57 条事件；五个技能各跨进程重放三次通过。
- 60 个候选请求：44 次成功、9 次失败、7 次拒绝，共 53 条执行记录；失败为 4 次 `no_progress` 和 5 次 `timeout`。四个 worker 耗时约 40.8 秒。
- 一步预算采集得到 6 次截断、3 次拒绝；事件时间、样本数、来源/输出 hash、原动作候选逐字段重现和标签掩码审查通过。

服务器数据根为 `/mnt/yuanyue/data/n1-events-uDR6Is/`，含 transitions、snapshots、candidates、budget-check 及对应日志。同步前备份为 `/mnt/yuanyue/backups/n1-before-events-S8Lqwp/source.tar.gz`。没有覆盖旧 pilot 或无关资产改动，也没有重新训练 actor。

本地与服务器的源技能结果和候选计数不完全相同；这里验证的是每个环境内从同一起点的重放一致性，不宣称跨硬件的逐位相同。两组样本都只用于标签链路验证，仍是固定场景族。

## 7. 下一步

后续参数化通道首步已实现并验证，见 [N2 参数化通道](N2_PASSAGE_SCENES_CN.md)。接下来推进几何约束候选、能力 profile 和多布局族；逐物理步接触覆盖、更完整的失稳判据、支撑关系和多步可达性仍需单独定义和验证。本节 N1 事件代码已发布为 `9ac5b07`，对应旧归档应使用该版本。服务器 Git HEAD 仍保留旧基线及工作树改动，不能直接用其 HEAD 代替数据中的运行代码指纹。