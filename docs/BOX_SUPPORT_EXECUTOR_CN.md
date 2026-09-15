# 移动支撑技能的正式执行与数据链路

更新日期：2026-09-15。用户已明确接受当前29/32组合技能成功率，要求利用现有技能推进项目。本轮完成共享executor、规则支撑规划、技能记录与快照接入，未重新训练或替换三份选定策略。集成基于此前发布的324b7a7，当前发布提交以实际Git历史为准；运行源码以记录和快照中的SHA-256为准。

本页的集成阶段已发布为f311203。其后有界几何目标和42请求物理分支pilot已完成，最新范围见 [几何候选记录](BOX_SUPPORT_GEOMETRY_CN.md)；下表保留f311203阶段的验证结果，旧归档仍要求原代码指纹。

## 1. 执行范围

[run_box_support_executor.py](../mujoco/run_box_support_executor.py) 使用统一AVX2数值入口，创建 [BoxSupportEnv](../mujoco/reconfigurable_navigation/box_support_env.py)、[BoxSupportPlanner](../mujoco/reconfigurable_navigation/box_support_planner.py) 和共享 [ReconfigurableExecutor](../mujoco/reconfigurable_navigation/runtime/executor.py)。

正式技能序列是 `PUSH -> NAV -> CLIMB -> NAV -> CLIMB -> STOP`。executor每次只执行最新计划的首个技能，随后重新观察并规划；接近位姿跟随当前箱面。PUSH包含收臂退出，CLIMB包含真实站稳准备和顶面NAV定位，保持原组合的动作、接触和停稳判据。

[BoxSupportBackend](../mujoco/reconfigurable_navigation/runtime/box_support_backend.py) 复用 [原组合控制函数](../mujoco/reconfigurable_navigation/box_support_control.py)。全程只有一次物理初始化，技能切换继承延迟队列，不改写机器人或箱体位姿。原有低摩擦通道与固定台阶课程继续使用原后端。

三份选定包分别是 `push-height020-stop200-bundle`、`climb-prepared-ground499-bundle`、`climb-prepared-gaps399-bundle`，必须保留226->12加IK及253->18的不同契约。策略和历史验收hash由 `continuous-climb-validation-v2.json` 绑定；该历史索引不改写为本轮结果。

## 2. 快照与标签

`--record-jsonl` 使用现有schema v2转换记录，包含前后观测、动作、控制步事件、耗时和终止原因。`--snapshot-dir` 在每个技能执行前保存完整N1快照，并写入相对路径与文件hash。

新快照额外保存Hybrid PUSH的机械臂参考姿态、接触保持状态、历史观察和动作、重力计算缓存，以及两段CLIMB的策略、延迟队列、当前物理runtime和规则规划进度。恢复后共享的环境、控制器和已完成操作集合仍指向同一分支对象；分支与现场物理状态相互独立。三份Box策略都包含在归档中，无需重新读取其外部包；NAV辅助定位仍使用仓库默认NAV策略，其文件hash纳入快照代码指纹。

快照只承诺技能起点恢复，不保存执行中的Python调用栈；带活动runtime回调的捕获会被拒绝。预算截断不产生物理失败标签，也不提供中途后继快照。非法或未支持的动作在物理执行前拒绝。加载仍要求 `--trusted`，并校验代码、运行库、CPU内核配置、线程数及成员hash；旧数据应使用原代码指纹，不能改manifest跳过检查。

## 3. 本轮验证

| 检查 | 结果与范围 |
| --- | --- |
| executor成功与失败代表回归 | seed500/509成功，510保留高台base_contact失败；阶段报告、耗时和末状态与原验收逐项相同 |
| 带快照的完整执行 | seed509/510，10条转换和10个快照；每回合物理初始化1次、技能切换reset0次 |
| 跨进程技能重放 | 全部10条，各1次，所有转换字段零容差一致，含NAV、PUSH、两段CLIMB和高台失败 |
| 跨机器技能重放 | 服务器加载本地归档，PUSH、登箱、高台成功/失败4条代表记录全部通过，绝对容差1e-6 |
| 新后端快照契约 | 本地、服务器各4项通过，覆盖控制器引用、策略独立性、IK/延迟恢复、拒绝与预算截断 |
| 既有接口回归 | 本地旧快照10项、后端4项通过；旧复杂课程seed0成功，1次reset |
| 现有候选采集器 | 4个代表起点、每点3请求：3成功、1物理失败、8前置拒绝 |

这里复用了已知验收种子，验证的是集成与数据一致性，不能据此重新宣称未见分布成功率。29/32仍是此前完整固定布局的组合结果；高台actor原生21/32、22/32的限制保留。

本地产物根目录：`/home/yuanyue/re-nav/artifacts/box-executor-integration-20260915/`。主要文件为 `recorded-509-510.json`、`transitions.jsonl`、`snapshots/`、`candidates/manifest.json` 和 `candidates/candidates.jsonl`。十个快照约1GB，包含重复的模型与策略，尚未做共享资产去重。数据不纳入Git，也不属于正式训练/验证/测试划分。

服务器数据根为 `/mnt/yuanyue/data/box-executor-integration-20260915/`，同步已用内容校验补齐，`cross-machine-replay.log` 保存跨机器重放结果。16份新增或修改源码已逐文件核对hash；覆盖前确认远端已有文件与发布基线或本地版本一致，并备份到 `/mnt/yuanyue/backups/box-executor-QPXP4C/source-before.tar.gz`。未pull/reset远端工作树，服务器Git HEAD仍不能代表实际同步源码。

## 4. 复现

在仓库根目录，使用已安装的MuJoCo专用环境。下面是本地路径；服务器对应解释器为 `/mnt/yuanyue/envs/go2-mujoco/bin/python`，模型包根为 `/mnt/yuanyue/data/box-skills-eval`。

```bash
export GO2_PYTHON=/home/yuanyue/re-nav/.envs/go2-arx5-nav/bin/python
export BOX_BUNDLES=/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14
export BOX_OUTPUT=$(mktemp -d /home/yuanyue/re-nav/artifacts/box-executor-XXXXXX)
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1

"$GO2_PYTHON" mujoco/run_box_support_executor.py \
  --push-policy "$BOX_BUNDLES/push-height020-stop200-bundle/policy.pt" \
  --climb-policy "$BOX_BUNDLES/climb-prepared-ground499-bundle/policy.pt" \
  --platform-policy "$BOX_BUNDLES/climb-prepared-gaps399-bundle/policy.pt" \
  --seed-offset 509 --seeds 2 \
  --reference-json "$BOX_BUNDLES/box-sequence-final-local500.json" \
  --output-json "$BOX_OUTPUT/report.json" \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --snapshot-dir "$BOX_OUTPUT/snapshots"

"$GO2_PYTHON" mujoco/replay_skill_records.py \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --indices 0 1 2 3 4 5 6 7 8 9 --repeats 1 --atol 0 --trusted

"$GO2_PYTHON" mujoco/collect_skill_candidates.py \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --output-dir "$BOX_OUTPUT/candidates" --indices 0 2 4 9 \
  --candidates-per-snapshot 3 --workers 2 --max-control-steps 5000 --trusted
```

输出文件和快照目录必须是新路径。提供 `--reference-json` 时，退出码检查是否重现参考结果，所以保留已知失败也可以通过集成回归；不提供参考时，只有请求的所有回合成功才返回0。报告始终保留真实成功数和失败原因。

## 5. 下一项

当前场景为5kg、摩擦0.4、20cm箱至40cm高台的固定布局。f311203阶段的12候选中8次拒绝来自固定目标限制，不能解释为物理不可达。后续版本已经支持有界停车、接近和落点，并实际执行了30个非默认候选；范围与数据见 [几何候选记录](BOX_SUPPORT_GEOMETRY_CN.md)。PUSH仍限于原正X推面，在线Oracle仍使用规则。

下一项是参数化多布局、后续任务评估和场景族划分，保留成功、物理失败、前置拒绝和截断。之后训练技能级状态转移、成功率、碰撞风险及代价模型，让模型参与候选选择。继续提高低层成功率不再是前置条件，VLM、RGB-D、JUMP与实机迁移后置。