# 参数化支撑布局与后续任务标签

日期：2026-09-15。本阶段基于已发布9c069e4开发，最新发布以实际Git历史为准。三份Box策略及29/32组合基线继续冻结，没有重新训练或调整成功判据。本页记录参数化布局、场景族分组和独立规则后续评估的数据阶段，当时尚未训练世界模型。

本文保留参数化数据阶段的结果；后续已经扩样并训练首个离线MLP，目标架构和评估结果见 [世界模型基线](WORLD_MODEL_BASELINE_CN.md)。该模型尚不能替换规则Oracle，本页21请求索引和诊断不回填新实验结果。

## 1. 场景配置

[BoxSupportScene](../mujoco/reconfigurable_navigation/box_support_scene.py) 采用可序列化配置，包含机器人/箱体/平台的XY和yaw、箱体与平台尺寸、箱体质量及摩擦。尺寸和材质在MjSpec编译前应用；位姿扰动只发生在物理初始化阶段，技能切换不reset。平台yaw当前必须为0，保留原正X推面限制。

配置、稳定scene ID和分组随快照保存；目标高度从真实平台几何读取，平台接近阶段使用实际箱顶高度。默认配置的模型数组完全一致，默认seed509/510的阶段报告、耗时和末位置也与原冻结验收完全一致。

`--scene-json` 接受配置对象，未提供字段使用默认值；`--scene-suite` 使用六个预定义布局。两个选项互斥，参数化布局不能与旧固定布局的 `--reference-json` 混用。

| 布局 | 平台变化 | 预定分组 | seed509原任务 |
| --- | --- | --- | --- |
| aligned | 默认 | train | 成功 |
| aligned_far | X=3.38m | train | 成功 |
| left | Y=0.04m | validation | 成功 |
| left_far_narrow | X=3.36m、Y=0.06m、宽1.5m | validation | PUSH超时 |
| right | Y=-0.04m | test | 成功 |
| right_near_wide | X=3.24m、Y=-0.06m、宽1.7m | test | 成功 |

场景族由平台相对箱体的横向偏置确定，分组在执行前固定。采集器拒绝同一族或组跨集合，也拒绝混合已分组和未分组的来源记录。这只是小规模分组设计，不构成训练完成或泛化验证。

## 2. 后续评估语义

[evaluate_suffix](../mujoco/reconfigurable_navigation/data/suffix.py) 从候选的真实终态快照创建独立分支，逐技能运行规则Oracle，遇到首个技能失败停止。它不重新做启动稳定或物理reset，并显式限制控制步数和技能数。原分支的物理状态及全局随机状态保持独立。

启用 `--evaluate-suffix` 后，标签写入每条候选的 `continuation`：

| 字段 | 语义 |
| --- | --- |
| oracle_task_success | 指定规则策略及预算下的任务结果；成功true，已终止失败false，拒绝或截断null |
| reachability | 只有实际到达目标时为true，其余为null；不提供全局不可达证明 |
| total_sim_time | 候选及后续尝试消耗的仿真时间，包含收臂、对位、准备和稳定；失败时是停止前耗时，不是到达目标的最优代价 |
| label_validity | 后续结果和代价各自的有效性，不能沿用即时技能的mask |
| cost_is_lower_bound | 截断时只观察到一段耗时，不可作为完整任务代价监督 |
| suffix_records | 后续技能、目标、状态、耗时和失败原因；不等同于每一步完整训练观测 |

候选预算与后续预算分别记录。即时 `skill_success` 继续只表示当前技能；成功PUSH可以对应失败的后续任务。规则拒绝或单次物理失败都不能证明其他候选或策略不存在成功路径。

## 3. 验证与数据

39项相关检查通过：32项场景/分组/候选/控制/后续语义检查，以及7项快照与目标路由检查。默认509/510与冻结基线逐项一致；六布局5/6完成，产生26条转换及6个首技能快照。

主数据另补入对齐布局seed510的后段失败案例，合计六布局、七回合、31条来源转换、7个首技能快照。主索引为21请求：

| 标签层 | 成功 | 失败 | 拒绝 |
| --- | --- | --- | --- |
| 候选技能 | 11 | 3 | 7 |
| 候选加规则后续任务 | 10 | 4 | 7 |

train/validation/test分别9/6/6条。候选ID唯一，场景与快照没有跨集合；来源、生成器、后续评估器和数据文件hash审计通过。七个原动作的整任务结果和总耗时与原回合一致。

seed510专项中，默认PUSH成功后高台发生base_contact，总尝试耗时21.12s；几何停车候选的PUSH同样成功，后续实际完成任务，耗时29.92s。两者的即时技能标签相同，整任务标签不同。该单例没有重估整体29/32成功率，也没有接入在线候选选择。

预算诊断单独保存：两个PUSH成功后只允许一个后续控制步，各消耗0.02s，整任务结果与可达性均为null，代价仅为下界；另有一次非法请求。诊断数据未纳入主索引的21请求。

本地根目录 `/home/yuanyue/re-nav/artifacts/box-layouts-20260915/`，服务器根目录 `/mnt/yuanyue/data/box-layouts-20260915/`。入口为 `dataset-index.json`，主分片是 `candidates/` 与 `failure-case/candidates/`，诊断是 `budget-check/`。本地保存报告、JSONL和索引；完整快照只保留在服务器，不纳入Git。

12份源码已备份同步并核对hash，覆盖前备份为 `/mnt/yuanyue/backups/box-layouts-XTwIgX/source-before.tar.gz`。服务器Git HEAD仍不代表实际同步源码，应按报告、采集器和快照指纹复现。旧归档不回填新标签，也不改manifest绕过指纹。

## 4. 运行入口

在服务器仓库根目录使用MuJoCo专用环境。场景评估器在有物理失败时保留退出码1，报告仍完整保存；候选采集器完成写入后返回0，实际成功、失败和未知计数保存在manifest。

```bash
export GO2_PYTHON=/mnt/yuanyue/envs/go2-mujoco/bin/python
export BOX_BUNDLES=/mnt/yuanyue/data/box-skills-eval
export BOX_OUTPUT=$(mktemp -d /mnt/yuanyue/data/box-layouts-XXXXXX)
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1

"$GO2_PYTHON" mujoco/run_box_support_executor.py \
  --push-policy "$BOX_BUNDLES/push-height020-stop200-bundle/policy.pt" \
  --climb-policy "$BOX_BUNDLES/climb-prepared-ground499-bundle/policy.pt" \
  --platform-policy "$BOX_BUNDLES/climb-prepared-gaps399-bundle/policy.pt" \
  --scene-suite --seed-offset 509 --seeds 1 \
  --output-json "$BOX_OUTPUT/source-report.json" \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --snapshot-dir "$BOX_OUTPUT/snapshots" --snapshot-first-only

BOX_INDICES=$("$GO2_PYTHON" -c 'import json,sys; rows=[json.loads(line) for line in open(sys.argv[1])]; print(" ".join(str(index) for index,row in enumerate(rows) if "snapshot_file" in row["metadata"]))' "$BOX_OUTPUT/transitions.jsonl")

"$GO2_PYTHON" mujoco/collect_skill_candidates.py \
  --record-jsonl "$BOX_OUTPUT/transitions.jsonl" \
  --output-dir "$BOX_OUTPUT/candidates" --indices $BOX_INDICES \
  --candidates-per-snapshot 3 --workers 4 --seed 42 --max-control-steps 5000 \
  --evaluate-suffix --suffix-control-steps 3000 --suffix-max-skills 8 --trusted
```

`--snapshot-first-only` 只为每个回合的第一个已接受技能存档；后续记录的 `full_snapshot_available` 为false且不带旧快照引用。需要所有技能起点时去掉该选项。专项失败与预算诊断复用同一入口，分别使用seed510的对齐配置，以及 `--suffix-control-steps 1`。

## 5. 下一阶段

在现有族内增加布局、起点和候选样本，保持分组并扩充失败类型，然后建立技能级特权状态转移、成功率、任务价值和代价模型。当前21请求只验证管线，不能支撑泛化或校准结论。在线规划仍是规则Oracle，世界模型、VLM、RGB-D和JUMP尚未实现。