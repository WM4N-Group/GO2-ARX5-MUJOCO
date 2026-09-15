# 有界几何候选与物理分支数据

日期：2026-09-15。先将executor/N1集成发布为 `f3112030178e6d3e0d4aacacccce54f09aecf035`，随后利用已接受的三份策略实现本页功能。29/32组合技能基线继续冻结，本轮没有训练或替换actor。最新发布提交以Git历史为准，实验源码由报告和快照中的hash绑定。

## 1. 已实现范围

[box_support_geometry.py](../mujoco/reconfigurable_navigation/box_support_geometry.py) 给出当前控制器可接受的目标区域，[BoxSupportBackend](../mujoco/reconfigurable_navigation/runtime/box_support_backend.py) 将请求交给真实控制函数。几何边界是此版本的请求约束，物理结果仍由独立MuJoCo分支执行得到。

| 技能 | 请求范围与执行语义 |
| --- | --- |
| PUSH停车 | 平台前方名义控制目标的X偏移不超过0.04m，横向不超过0.08m并受平台宽度限制；只接受当前正X推面、零目标朝向和向前推动 |
| NAV接近 | 相对当前支撑入口的法向/切向偏移各不超过0.08m，朝向偏移不超过0.08rad；高台入口还需落在当前箱体的机器人支撑范围内 |
| CLIMB落点 | 顶面局部偏移不超过前后0.10m、横向0.08m，并受机器人尺寸与顶面边界限制；保持原目标高度和朝向 |

PUSH请求直接进入Hybrid控制器的目标，名义值保留原有0.02m控制偏置，最终箱体停放坐标从物理输出读取。NAV与CLIMB使用支撑面局部偏移，随着支撑体移动/转动重新计算世界目标。CLIMB控制和完成判据使用同一落点，原四足支撑、姿态、速度、距离容差及持续停稳条件保持不变。

零偏移沿用原执行路径，未通过关节或物体位姿改写实现候选目标。场景仍是自由5kg、摩擦0.4、20cm箱与40cm高台，尚未扩展成任意推面或参数化多布局。

## 2. 候选与记录

[build_candidates](../mujoco/reconfigurable_navigation/data/candidates.py) 对 `box_support_nominal` 使用真实快照观测生成可重复的几何候选，类型为 `support_parking`、`support_approach` 和 `support_landing`；保留原动作 `reference` 和主动构造的 `invalid_request`。其他场景继续使用原局部参数扰动。

[collect_skill_candidates.py](../mujoco/collect_skill_candidates.py) 复用N1的独立快照rollout。每个候选携带实际目标、来源快照、物理转换、成功/失败/拒绝、事件及耗时。几何候选改变了运行代码指纹，旧f311203快照需要原版本；本轮重新生成了匹配新代码的快照。

技能成功是单次动作的标签。后续任务可达性和完整任务价值尚未评估，相应标签仍为unknown，不能把短距离停车成功直接解释为后续登高台一定可行。

## 3. 验证结果

本地27项相关检查通过：几何6项、原候选接口9项、箱体控制契约6项、新快照/目标路由6项。服务器默认seed509成功、seed510保留高台机身接触失败，阶段报告、耗时和末位置均与冻结验收完全一致。

从新记录中选择索引0/1/2/3/4/9，共6个技能起点，分别覆盖PUSH、两段NAV、两段CLIMB及已知高台失败起点。每点7个请求，用4个CPU worker执行：

| 候选来源 | 成功 | 物理失败 | 拒绝 |
| --- | --- | --- | --- |
| 原动作reference | 5 | 1 | 0 |
| 停车候选 | 5 | 0 | 0 |
| 接近候选 | 10 | 0 | 0 |
| 落点候选 | 9 | 6 | 0 |
| 主动非法请求 | 0 | 0 | 6 |
| 总计 | 29 | 7 | 6 |

30个非默认几何候选全部实际执行，最终机器人位置与同一起点原动作均相差超过1e-6m。6个reference转换与原记录逐字段完全相同；7次物理失败均为 `base_contact`。来源JSONL、当前源码、候选文件hash和拒绝标签审计通过。

42请求共享6个起点、2个已知回合、1个固定布局族，属于 `pilot_unsplit`。它不是42个独立场景，也没有建立新的未见分布成功率；原29/32基线没有重估。

## 4. 产物与复现

服务器根目录：`/mnt/yuanyue/data/box-geometry-20260915/`。包含 `source-report.json`、`transitions.jsonl`、10个 `snapshots/*.snapshot` 和 `candidates/{manifest.json,candidates.jsonl}`。报告与JSONL已取回本地 `/home/yuanyue/re-nav/artifacts/box-geometry-20260915/`；约1GB完整快照保留在服务器，未纳入Git。

9份代码已备份同步并逐文件核对hash；备份为 `/mnt/yuanyue/backups/box-geometry-hh93sL/source-before.tar.gz`。服务器仍是旧Git checkout加同步源码，采集manifest的Git HEAD不能单独代表实际运行版本，应使用源码/采集器/候选生成器和快照指纹。

在服务器仓库根目录执行，输出目录必须是新目录：

```bash
export GO2_PYTHON=/mnt/yuanyue/envs/go2-mujoco/bin/python
export BOX_BUNDLES=/mnt/yuanyue/data/box-skills-eval
export BOX_OUTPUT=$(mktemp -d /mnt/yuanyue/data/box-geometry-XXXXXX)
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1

"$GO2_PYTHON" mujoco/run_box_support_executor.py \
  --push-policy "$BOX_BUNDLES/push-height020-stop200-bundle/policy.pt" \
  --climb-policy "$BOX_BUNDLES/climb-prepared-ground499-bundle/policy.pt" \
  --platform-policy "$BOX_BUNDLES/climb-prepared-gaps399-bundle/policy.pt" \
  --seed-offset 509 --seeds 2 \
  --reference-json "$BOX_BUNDLES/box-sequence-final-server500.json" \
  --output-json "$BOX_OUTPUT/source-report.json" \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --snapshot-dir "$BOX_OUTPUT/snapshots"

"$GO2_PYTHON" mujoco/collect_skill_candidates.py \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --output-dir "$BOX_OUTPUT/candidates" --indices 0 1 2 3 4 9 \
  --candidates-per-snapshot 7 --workers 4 --max-control-steps 5000 --seed 42 --trusted
```

## 5. 下一项

将当前候选用于参数化多布局，并增加带明确预算的后续任务评估和完整代价标签；所有来自同一场景族、回合与快照的分支应分组，避免跨训练/测试泄漏。之后建立技能级特权状态世界模型，再接入在线候选选择。

在线规划当前仍是规则Oracle，没有使用此次候选的物理评分来选动作。继续提高低层成功率不再是前置条件，VLM、RGB-D、JUMP和实机迁移后置。