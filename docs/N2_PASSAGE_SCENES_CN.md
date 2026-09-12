# N2 参数化通道与物理扫描

验证日期：2026-09-12。本轮在已发布的 N1 事件提交 `9ac5b07` 上继续开发，已同步并在 yuanyue 服务器验收。实现的是参数化 Blocked Passage 的第一步，不代表 N2 的几何候选、能力标定和多场景训练集全部完成。

## 1. 已实现范围

- [passage_scene.py](../mujoco/reconfigurable_navigation/passage_scene.py) 提供 `PassageScene`、可复现 `passage_sweep(seed)` 和 `ParameterizedPassageEnv`。
- 使用 MuJoCo `MjSpec` 读取原场景并在编译前修改墙体、箱体尺寸/质量/摩擦和目标标记，惯量由 MuJoCo 根据实际几何与质量计算。
- 箱体尺寸/质量、墙体尺寸/位置从实际 model/data 提取为真值，不再把旧场景常量用于新场景。
- [evaluate_passage_scenes.py](../mujoco/evaluate_passage_scenes.py) 使用生产 Oracle/executor 完成物理任务、事件记录、可选技能起点快照和任务汇总。
- 原有快照归档保留场景配置和随机数状态；候选输出继承场景参数、场景 ID、扫描分组与能力信息。
- 默认参数的初始 qpos、质量、惯量、几何、摩擦及静态 body 位置与旧通道一致；旧复杂课程和技能接口保留。

不会在执行技能时用 `set_box_pose()` 移动物体。配置摆放只发生在 episode 重置；每个任务只有一次物理 reset。既有 actor、控制阈值和 Oracle 技能选择规则没有因为参数扫描改变。

## 2. 参数与输入约束

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `corridor_width` | 1.6 | 墙体内表面之间的通道宽度，米 |
| `box_size` | [0.44, 1.44, 0.60] | 箱体三个局部轴的完整边长，米 |
| `box_mass` | 5.0 | 真实箱体质量，千克 |
| `box_friction` | 0.005 | 箱体滑动摩擦系数；保留 priority=1，扭转/滚动系数为 0.001/0.0001 |
| `robot_pose` | [-2.30, 0, 0] | 起点 [x,y,yaw]，米/弧度；base 初始高度保留 0.445 米 |
| `box_pose` | [0,0,0] | 箱体 [x,y,yaw]；底面放在地面，高度随尺寸变化 |
| `goal_pose` | [1.8,0,0] | 目标位置及保留的第三分量；当前平面 Oracle 使用 x/y，目标 NAV yaw 固定为 0 |
| `push_target_pose` | [2.45,0,0] | 当前人工配置的箱体目标 [x,y,yaw]；PUSH 主要跟踪平面位移 |

输入必须有限，尺寸/质量/宽度/摩擦必须为正；初始旋转箱体不能与通道墙体相交。重置后的机器人墙体/箱体接触也会被拒绝。这里没有提供完整任务可行性证明，合法输入仍可能在执行中失败。

`PassageScene` 的 JSON 参数可往返恢复，并由规范化参数计算 `scene_id`。扫描种子先生成与旧通道相同分布的起点和箱体位置，随后各案例只修改一项；显式配置的位姿不会在 reset 时被额外随机移动。当前通道长度和布局拓扑仍固定。

## 3. 使用方式

在服务器仓库根目录和隔离 MuJoCo 环境中执行，每次使用新的输出目录：

```bash
cd /mnt/yuanyue/GO2-ARX5-MUJOCO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GO2_PYTHON=/mnt/yuanyue/envs/go2-mujoco/bin/python
"$GO2_PYTHON" mujoco/check_passage_scenes.py -v
"$GO2_PYTHON" mujoco/evaluate_passage_scenes.py --output-dir /mnt/yuanyue/data/passage-sweep-new --seeds 1 --snapshots
```

可以用 `--cases baseline mass_low friction_high` 选择扫描项，或用 `--seed-offset` 指定起始种子。`--scene-json` 接受一个 `PassageScene` 参数对象，未给出的字段使用默认值；该选项与 `--cases` 互斥。

输出布局：

```text
output/
  manifest.json
  episodes.jsonl
  transitions.jsonl
  snapshots/<episode-id>-<skill-index>.snapshot
```

没有 `--snapshots` 时仅写 JSONL。`episodes.jsonl` 保留成功、规则拒绝和执行失败的任务结果；没有接受技能的任务不会伪造 transition。`succeeded` 是 executor 任务结果；`accepted` 还检查单次 reset、采样到的非法接触/无效状态及每个已执行 PUSH 的指尖接触。没有执行 PUSH 时接触字段是 null。

物理失败是扫描结果，命令完成不意味着每个案例成功；manifest 的 `complete=true` 只表示所有请求已处理。异常中止保留 `complete=false`，当前没有断点续采。完成时写入 episode/transition 文件 hash、运行代码指纹、版本和脚本 hash。

候选采集仍使用原命令，但现在会继承正确的通道元数据：

```bash
"$GO2_PYTHON" mujoco/collect_skill_candidates.py --record-jsonl /mnt/yuanyue/data/passage-sweep-new/transitions.jsonl --output-dir /mnt/yuanyue/data/passage-candidates-new --candidates-per-snapshot 3 --workers 4 --trusted
```

旧复杂课程记录根据 `scenario=complex_course` 兼容识别；未知来源必须显式提供 `scene_family`，不会默认冒充复杂课程。

## 4. 本次服务器结果

通过 7 项参数化场景合约、23 项事件/候选合约和 10 项真实快照合约，共 40 项。默认复杂课程 10/10、CLIMB 切换 10/10、PUSH 5/5 通过。新增场景的七项合约在本地也通过。

seed=0 的十个单变量案例结果如下：

| 案例 | 改动 | 实际结果 |
| --- | --- | --- |
| baseline | 原分布参数 | NAV-PUSH-NAV 成功 |
| mass_low | 箱体 3 kg | 成功 |
| mass_over_limit | 箱体 10 kg | 规则返回 blocking_object_too_heavy，0 条技能记录 |
| friction_high | 箱体摩擦 0.15 | 两次 PUSH contact_timeout，耗尽失败预算 |
| corridor_wide | 通道宽 1.8 m | 成功 |
| box_narrow | 箱体宽 1.2 m | 成功 |
| box_yaw | 箱体偏航 0.04 rad | 成功 |
| box_position | 箱体 x=0.15 m | 成功 |
| robot_start | 起点 x=-2.15 m | 成功 |
| goal_position | 目标 x=1.6 m | PUSH 成功后两次 NAV timeout，耗尽失败预算 |

总计 7/10 接受，产生 28 条真实技能记录及起点快照。失败没有改成“物理不可行”标签，也没有为了通过扫描而修改控制器、任务阈值或参数。

28 个起点各采集 3 个候选，总计 84 个请求：50 次成功、6 次执行失败、28 次前置拒绝，即 56 条执行记录。失败为 4 次 contact_timeout 和 2 次 NAV timeout。四个 worker 约 34.6 秒完成。

所有 28 个 reference 候选逐字段重现源记录。抽取默认、低质量、箱体偏航、PUSH 接触超时和 NAV 超时等九个起点，各在新进程中重放三次通过，容差 `1e-6`。文件 hash、模型/真值质量与尺寸、场景分组及候选来源元数据审查通过。

服务器产物根：`/mnt/yuanyue/data/n2-passages-aoEpYv/`，其中 `sweep/` 保存任务/技能记录和快照，`candidates/` 保存候选与 manifest，根目录保存执行日志。同步前源码备份：`/mnt/yuanyue/backups/n2-before-passages-M0ySP5/source.tar.gz`。

本地另有两个早期 CLI 验证案例，位于 `/home/yuanyue/re-nav/artifacts/n2-passages-6QAiDr/`；其中任务级接触汇总是修正空 PUSH 集合标签之前的探测产物，正式结果以上述服务器目录为准。

## 5. 数据边界与下一步

这些数据只有一个 `blocked_passage` 布局族、一个扫描种子。十组参数不是十个独立场景族。相同基础布局的物理/几何变体带相同 `sweep_group_id`，候选保留该分组；当前 manifest 明确为 unsplit 或 pilot_unsplit，没有生成 train/validation/test 划分。

能力 ID `go2_arx5_rule_baseline_v1` 仅标识冻结的规则约束，不代表新测得的最大推力、摩擦范围或台阶能力。没有因为 10 kg 被规则拒绝而宣称机器人在物理上绝对推不动，也没有把高摩擦接触超时等同于力不足。

新归档的代码指纹包含 `passage_scene.py`。此前 N1 事件归档需使用 `9ac5b07` 的对应代码，早期 N1 pilot 使用 `f857437`；不应绕过运行代码/版本校验。服务器仍保留旧 Git HEAD 和同步后的工作树，数据重放以实际运行代码指纹为准。

下一步先做推面、接近位姿和停放区域的几何候选，使用这批真实失败与成功起点比较多个候选；再扩展布局拓扑、能力 profile 和按场景族分组的评测。当前固定 push_target 仍是人工配置，尚无世界模型、学习型规划或视觉输入。