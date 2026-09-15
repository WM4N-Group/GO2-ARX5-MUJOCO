# GO2-ARX5 可重构导航原型

交接范围更新：2026-09-15。当前代码和任务以 [主交接](../../docs/AGENT_HANDOFF_CN.md) 为准，本文主要说明保留的生产Oracle、旧低摩擦通道及低台阶接口。用户已接受移动箱高台组合29/32；该流程通过Box后端接入共享executor和N1快照，使用 [正式CPU入口](../run_box_support_executor.py)，验收与数据见 [集成记录](../../docs/BOX_SUPPORT_EXECUTOR_CN.md)。历史独立流程与训练结果见 [箱体训练记录](../../docs/BOX_SKILL_TRAINING_CN.md)。

该目录实现世界模型项目前置的 Oracle-first 原型。当前阶段使用 MuJoCo 真值状态，不依赖 RGB-D、VLM 或学习世界模型，用于验证场景定义、结构化技能接口和可达性判断。

新Box后端已支持有界停车、入口和顶面目标，现有候选采集器完成42请求的真实物理分支评估，见 [几何候选记录](../../docs/BOX_SUPPORT_GEOMETRY_CN.md)。这仍是固定布局离线pilot，在线规划尚未使用候选物理评分。

后续版本还支持参数化六布局及带预算的规则后续评估，见 [场景族与任务标签](../../docs/BOX_SUPPORT_LAYOUTS_CN.md)。在线规划仍为规则Oracle，后续评估只在独立离线分支中执行。

首个离线位姿MLP已经训练，代码在 [world_model](world_model/)，框架与未通过的规划质量对照见 [世界模型基线](../../docs/WORLD_MODEL_BASELINE_CN.md)。目标对象Transformer ensemble尚未实现，当前执行器未使用学习模型选动作。

## 保留的生产原型功能

- GO2-ARX5 Blocked Passage 专用 MuJoCo 场景。
- 带自由关节、质量和摩擦的动态箱体。
- 结构化 `Observation`、`Capability`、`ObjectState` 和 `SkillAction`。
- 机器人 footprint 膨胀的二维占据栅格。
- 支持八邻域、禁止穿角的 A* 路径规划。
- 通过移除单个可移动物体识别阻挡物。
- Oracle 生成 `NAV -> PUSH -> NAV -> STOP` 技能序列。
- 根据 `max_pushable_mass` 拒绝能力不足的方案。
- 统一 NAV、PUSH、CLIMB Skill 生命周期。
- 将世界坐标误差转换为策略使用的机体系 `vx/vy/yaw_rate`。
- 支持 NAV actor 与独立 CLIMB actor 在同一 MuJoCo 状态中无 reset 切换。

## 运行自检

先激活已安装的MuJoCo专用环境，并在仓库根目录执行下列命令。本地环境为 `/home/yuanyue/re-nav/.envs/go2-arx5-nav`，服务器为 `/mnt/yuanyue/envs/go2-mujoco`；未激活时可使用对应解释器完整路径或Micromamba `run -p`。不要再假定存在旧名为 `leggedmanip` 的Conda环境，也不要在Isaac环境里替换CPU版PyTorch。

```bash
python mujoco/check_reconfigurable_navigation.py
```

指定随机场景数量：

```bash
python mujoco/check_reconfigurable_navigation.py --seeds 500
```

检查内容包括：

1. 箱体位于通道中时，目标不可直接到达。
2. 规划器正确识别阻挡箱体 `object_id=10`。
3. 生成 `NAV -> PUSH -> NAV -> STOP` 结构化计划。
4. Oracle 设置推箱结果后，目标变为可达。
5. 箱体质量超过 Capability 时，规划器拒绝执行。
6. NAV 输出满足速度上限，并在到达目标位姿后停止。

## Oracle 规划可视化

循环显示随机场景：

```bash
python mujoco/visualize_reconfigurable_navigation.py
```

只播放一次，或提高动画速度：

```bash
python mujoco/visualize_reconfigurable_navigation.py --once
python mujoco/visualize_reconfigurable_navigation.py --speed 2
```

Viewer 将依次显示机器人接近箱体、将橙色箱体推出阻挡区域、重新规划并到达绿色目标点。终端同步打印当前规划和执行阶段。关闭 Viewer 窗口即可停止循环。

该动画是 Oracle 规划和场景重构的运动学预览，机器人与箱体位姿由演示器逐帧设置；它不表示底层策略已经完成真实接触推动。

## 真实动力学 NAV 可视化

以下入口不再修改机器人基座位姿。它构造 210 维历史观测，运行现有 `210 -> 18` TorchScript 策略，通过 PD 力矩控制和 `mujoco.mj_step()` 产生真实步态：

```bash
python mujoco/visualize_policy_navigation.py
```

无 Viewer 快速诊断：

```bash
python mujoco/visualize_policy_navigation.py --headless --no-realtime
```

批量运行随机场景回归，并按 NAV 成功率门槛返回退出码：

```bash
python mujoco/evaluate_policy_navigation.py
```

当前默认部署策略与上游 `zzzJie-Robot/LeggedManip_Lab` 发布的 GO2-ARX5 `policy.pt` 逐字节一致。运行时保持上游 Sim-to-Sim 契约，包括 210 维三帧历史观测、Isaac/MuJoCo 关节映射、动作范围、PD 控制和启动稳定阶段。目标点 NAV 已在本地通过 100 个、远端通过 20 个随机场景回归，成功率均为 `100%`，满足设计方案中 `NAV >= 95%` 的验收线。

## 真实动力学 Blocked Passage

以下入口执行完整的真实动力学 `NAV -> PUSH -> NAV -> STOP`。机器人和箱体位姿不由演示器直接修改；GO2-ARX5 策略通过 PD 力矩和 `mujoco.mj_step()` 产生运动，PUSH 必须先检测到 ARX5 指尖接触才进入推动阶段：

```bash
python mujoco/visualize_policy_reconfigurable_navigation.py
```

无 Viewer 快速运行：

```bash
python mujoco/visualize_policy_reconfigurable_navigation.py --headless --no-realtime
```

批量回归：

```bash
python mujoco/check_push_skill.py --seeds 20
```

当前固定 MVP 分布在本地通过 `20/20`、远端通过 `5/5` 完整任务回归。每个成功 episode 都包含指尖接触、物理箱体位移、重构后路径可达和最终 NAV 到达，且没有机身接触或非法碰撞。机械臂与箱体的最大接触穿透为 `3.9-7.0 mm`，低于 `10 mm` 验收上限。PUSH 状态机依次执行 `ALIGN -> CONTACT -> PUSH -> VERIFY -> RETREAT`。统一 executor 每次只执行最新计划的首个技能；典型 episode 产生 4 次重规划：`NAV -> PUSH -> NAV -> STOP`。

## CLIMB 与无 reset 技能切换

独立 CLIMB Viewer：

```bash
python mujoco/visualize_climb_policy.py --policy mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt --duration 20
```

通过真实 executor 检查 `NAV -> CLIMB -> NAV`：

```bash
python mujoco/check_climb_skill_switching.py --seeds 10
```

当前回归为 `10/10`。每个 episode 只调用一次物理 reset；技能切换仅初始化 actor 内部 history、last action 和 delayed target buffer。receding-horizon executor 可以追加连续同类 NAV 修正，但压缩后的阶段必须是 `NAV -> CLIMB -> NAV`，且每条技能记录都成功。

## PUSH + CLIMB 复杂课程

生产 `OraclePlanner` 现在可从 `PLATFORM` 的显式 entry/landing portal 生成 CLIMB，并能识别阻挡平台入口的可移动箱体。完整结构计划为：

```text
NAV -> PUSH -> NAV -> CLIMB -> NAV -> STOP
```

真实物理批量验收：

```bash
python mujoco/check_complex_course.py --seeds 10
```

可视化同一执行路径：

```bash
python mujoco/check_complex_course.py --seeds 1 --visualize
```

当前复杂课程通过 `10/10`。每个 episode 都有 6 次逐技能重规划、一次物理 reset、真实指尖推动、零机身接触、零非法碰撞，并最终在顶层平台到达目标。NAV 切换到 CLIMB 前有显式 `0.5 s` 默认姿态准备阶段，避免机械臂遗留姿态污染 CLIMB 启动状态。

## 技能边界数据记录

在已激活 MuJoCo 环境的仓库根目录中运行：

```bash
python mujoco/check_skill_transitions.py -v
python mujoco/check_complex_course.py --seeds 10 --record-jsonl logs/n1-transitions.jsonl
```

`--record-jsonl` 是可选的单进程追加输出，不改变默认技能执行或任务成功条件。每条数据包含 schema_version、动作、前后观测拷贝、前序技能、技能状态、仿真时间和 episode/seed 元数据；总耗时包含 CLIMB 准备阶段。用户中断的 skill_success 为 null，非有限数值不会作为合法 JSON 写出。

当前 schema v2 增加控制步过程事件、PUSH 阶段、结束原因与标签掩码；中途出现后恢复的接触/异常仍会保留。事件只覆盖控制边界，不保证捕获物理子步短暂接触。新版分别在本地和服务器通过 32 项合约和 60 请求物理候选验证，完整语义见 [过程事件记录](../../docs/N1_SKILL_EVENTS_CN.md)。重放新版应重新生成快照，不绕过旧归档的运行代码指纹检查。

新服务器已通过 7 项合约测试及带记录复杂课程 `10/10`，生成 50 条记录，其中 48 条技能成功、2 条失败。整体到达目标不等于每个技能都成功，不能把任务结果覆盖为每条技能的正标签。

单独的 JSONL 是技能边界观测记录，完整物理和控制器快照需要额外指定 `--snapshot-dir`。并行 worker 应写不同文件；异常中止后应检查末行完整性。详细新机结果见 [部署与开发记录](../../docs/YUANYUE_SERVER_STATUS_CN.md)。

### 完整快照与物理候选

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python mujoco/check_complex_course.py --seeds 1 --record-jsonl logs/replay-source.jsonl --snapshot-dir logs/replay-snapshots
python mujoco/replay_skill_records.py --record-jsonl logs/replay-source.jsonl --indices 0 1 2 3 4 --repeats 3 --trusted
python mujoco/collect_skill_candidates.py --record-jsonl logs/replay-source.jsonl --output-dir logs/candidate-pilot --candidates-per-snapshot 12 --workers 4 --trusted
```

新机已验证 23 项合约测试和 50 个原动作的复现，并生成 600 个候选请求，其中 542 条实际执行、58 条前置拒绝。完整数据分布、版本/信任限制和命令见 [N1 快照与数据说明](../../docs/N1_REPLAY_DATA_CN.md)。这些是固定场景族的 pilot，不是已完成泛化评测或世界模型训练。

## 参数化通道扫描

`PassageScene` 支持箱体质量/摩擦/尺寸/位姿、通道宽度、机器人起点和目标参数；MuJoCo 模型与真值观测同步变化，并进入完整快照。参数化扫描使用原 Oracle/executor，实际失败和规则拒绝都作为数据保留。

```bash
python mujoco/check_passage_scenes.py -v
python mujoco/evaluate_passage_scenes.py --output-dir logs/passage-sweep-new --seeds 1 --snapshots
python mujoco/collect_skill_candidates.py --record-jsonl logs/passage-sweep-new/transitions.jsonl --output-dir logs/passage-candidates-new --candidates-per-snapshot 3 --workers 4 --trusted
```

每次使用新输出目录；候选继承 scene_family、scene_id、scene_parameters 和 sweep_group_id。服务器 40 项合约通过，十类单变量扫描 7/10 接受，84 个候选为 50 成功、6 失败、28 拒绝，详见 [N2 参数化通道](../../docs/N2_PASSAGE_SCENES_CN.md)。这是一个通道布局族的扫描，没有完成几何 grounding、能力标定或训练/测试划分。

## 模块

| 文件 | 作用 |
| --- | --- |
| `representations.py` | 观测、对象、能力和技能动作类型 |
| `env.py` | MuJoCo 场景加载、随机化与真值观测 |
| `occupancy.py` | 占据栅格、障碍膨胀和 A* |
| `oracle_planner.py` | 阻挡物识别与规则式技能规划 |
| `locomotion_runtime.py` | 210 维观测、TorchScript 推理、PD 控制和 MuJoCo 动力学 |
| `climb_runtime.py` | 253 维 CLIMB 观测、DelayedPD 语义和共享 MuJoCo 状态控制 |
| `complex_course_env.py` | PUSH 后转入 90 度楼梯的复杂课程真值观测 |
| `passage_scene.py` | 参数化通道、单变量扫描及物理一致真值 |
| `evaluate_passage_scenes.py` / `check_passage_scenes.py` | 参数扫描、场景合约、数据与快照入口 |
| `skills/base.py` | 统一技能生命周期和低层命令类型 |
| `skills/navigate.py` | NAV 目标跟踪和速度命令生成 |
| `skills/push.py` | 带接触、进度、安全间距和退出检测的机械臂 PUSH 状态机 |
| `skills/climb.py` | CLIMB 进入条件、平台位置/高度完成条件和超时控制 |
| `runtime/safety.py` | 观察有效性、非法碰撞和执行预算检查 |
| `runtime/replanner.py` | 将每次最新 Oracle 计划归约为一个下一动作 |
| `runtime/executor.py` | 逐技能执行、重新观测、失败重试和终态控制 |
| `data/transition.py` | 可序列化技能边界观测、时间和中断语义 |
| `data/events.py` | 控制步接触/状态事件边沿与 PUSH 阶段记录 |
| `data/snapshot.py` / `data/snapshot_io.py` | 独立物理快照、原生缓存归档与技能级回放 |
| `data/candidates.py` | 采集用参数扰动与拒绝/截断标签 |
| `replay_skill_records.py` / `collect_skill_candidates.py` | 归档重放与批量物理候选采集入口 |
| `check_skill_transitions.py` | 记录拷贝、标签、准备时长和默认行为合约检查 |
| `evaluate_policy_navigation.py` | 多随机场景 NAV 成功率回归 |
| `check_push_skill.py` | 真实 NAV-PUSH-NAV 完整任务回归 |
| `check_climb_skill_switching.py` | 同一物理状态中的 NAV-CLIMB-NAV 切换回归 |
| `check_complex_course.py` | 生产 planner 驱动的 PUSH+CLIMB 完整物理回归与 Viewer |
| `visualize_policy_reconfigurable_navigation.py` | 真实物理 Blocked Passage 可视化 |
| `blocked_passage.xml` | Blocked Passage 场景 |
| `complex_course.xml` | 狭窄通道 PUSH 和旋转楼梯 CLIMB 组合场景 |

## 旧生产原型边界

运动学 Oracle 动画仍使用 `set_box_pose()` 展示反事实，但真实物理入口和回归不调用它。当前 PUSH 使用 ARX5 各 link 的 mesh convex hull 防止穿模，并将任何机身或腿部接触箱体视为失败。5 kg 箱体被明确建模为高优先级、摩擦系数 `0.005` 的脚轮箱；它不代表当前策略能用机械臂推动普通高摩擦 5 kg 箱体。PLATFORM 当前依赖场景提供显式 entry/landing portal，尚未从任意网格自动提取；DelayedPD 仍采用已验证的固定 delay profile。当前可执行技能是 NAV、PUSH、CLIMB；JUMP 只有枚举和设计文档占位，尚无 actor、runtime 或 `JumpSkill`。恢复站立、接触重建和更宽参数随机化仍属于后续工作。
