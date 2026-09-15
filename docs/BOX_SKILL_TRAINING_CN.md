# 正常摩擦推箱与低箱攀爬训练

交接范围更新：2026-09-15；本文件按阶段保留2026-09-12至2026-09-14实验记录，数值结果不是本次文档整理重新运行所得。代码及当时文档已随 `008a7958cf542fe05a318290d4534358e0704f2b` 发布；当前机器、发布范围与待办以 [主交接](AGENT_HANDOFF_CN.md) 为统一入口，服务器版本见 [部署状态](YUANYUE_SERVER_STATUS_CN.md)。新权重、起步数据和录像不在Git，模型包和文件名见本文件末尾的最终组合记录。

用户在 2026-09-12 明确将优先级调整为先训练可靠的 PUSH 和登箱 CLIMB，再扩大世界模型数据。2026-09-14 的 20 cm 箱体 PUSH 已在 5 kg、摩擦 0.4、60 cm 目标下通过两组独立 Isaac 验收，各 31/32；资产和执行器契约修正后，MuJoCo 两组各 32/32。真实移动箱到 40 cm 高台的初版完整流程为 18/32；连续起步、完整场景及间隙专训后，最终新种子 500-531 在本地和服务器均为 29/32，逐回合结局一致。成功率包括 NAV 对位及顶面定位，高台 actor 单独原生验收仍未达标；剩余3次失败均为高台机身接触。当前训练均已结束，旧接受权重和生产 Oracle/executor 未替换。

最新方向：用户已接受29/32组合水平，三份策略冻结，继续提高低层成功率不再是推进前置条件。共享executor通过独立Box后端接入了该组合，并支持技能起点快照与N1重放，详见 [正式执行与数据记录](BOX_SUPPORT_EXECUTOR_CN.md)。本文件后续按时间保留训练历史，早期“尚未组合”“继续训练”的安排不覆盖此项决定。

## 1. 目标与物理假设

- PUSH：推动带实际滑动阻力的箱体，而不是仅验证摩擦系数 0.005 的脚轮箱基线。首轮工程范围为质量 3-8 kg、滑动摩擦系数 0.2-0.6，静摩擦不低于动摩擦，恢复系数 0。
- CLIMB：从地面登上 20-30 cm 的单箱，再能够从移动到位的箱体登上更高平台；原每级 2-12 cm 的连续台阶训练不能覆盖这一目标。
- 箱体首版顶面为 1.2 x 1.2 m，PUSH 高度 0.25 m、目标平移 0.60 m。它是明确的训练假设，不代表任意尺寸或材质都已验证。
- 当前通过改变接触材质表达摩擦，箱体 linear/angular damping 均为 0；没有将刚体阻尼或机械臂关节阻尼混作滑动摩擦。
- 新机器人 profile 明确设置腿部力矩 23.7/23.7/45.43 N.m、机械臂 20/20/20/7/5/5 N.m；腿部速度 30.1/30.1/15.7 rad/s，机械臂采用 3 rad/s 的工程限制。后者不是实机速度规格的测量结果。

正式验收以关闭探索噪声、固定每环境回合数的数据为准。首轮名义目标为 5 kg/摩擦 0.4 的 PUSH 至少 29/32 成功；原 CLIMB 计划在 20/25/30 cm 各至少 29/32，用户随后接受当前 20 cm 阶段，故冻结该 actor，暂不继续优先训练 25/30 cm。质量/摩擦边界与箱体尺寸变化分别报告，不能根据名义点推断整个范围。之后必须验证可移动支撑箱与箱体到高平台的切换，才可声称最终组合场景可用。

## 2. 实现入口

| 文件 | 作用 |
| --- | --- |
| [box_push_env_cfg.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/box_push_env_cfg.py) | 真实刚体箱、质量/摩擦随机化、接触传感器、224 维 PUSH actor 输入 |
| [box_push.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_push.py) | link0 坐标末端命令、箱体状态、有效接触推进和停稳判定 |
| [box_climb_env_cfg.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/box_climb_env_cfg.py) | 单箱高度课程、253 维 CLIMB 输入、停稳与姿态要求 |
| [box_climb.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_climb.py) | 箱顶目标速度、真实脚底支撑、支撑高度相关机身净空和课程 |
| [box_robot_cfg.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/box_robot_cfg.py) | 新任务专用的显式 PD/求解器限制，不改旧资产 profile |
| [box_rewards.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_rewards.py) | 有效完成奖励、目标方向收益和顶面支撑判断 |
| [initialization.py](../scripts/rsl_rl/initialization.py) | 已有权重热启动、直接 ELU actor 恢复 |
| [evaluate_box_skills.py](../scripts/evaluate_box_skills.py) | 关闭探索噪声的固定回合评估、失败原因、物理参数和权重 hash |
| [box_push_actions.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_push_actions.py) | 混合 PUSH 的 12 维腿部 actor 与 18 维实际命令路由 |
| [box_push_ik.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/box_push_ik.py) | 限速微分 IK、重力补偿、接触准备与手臂姿态保持 |
| [export_box_push_actor.py](../scripts/export_box_push_actor.py) | 验收门禁、TorchScript 输出对照、训练参数和评估 manifest |

新任务名：`GO2-ARX5-Box-Push`、`GO2-ARX5-Box-Climb`，各有 `-Play`。PUSH 在原 NAV 210 维历史观测之后加入 14 维箱体状态，新增首层输入权重置零；CLIMB 保留原 253 维接口。两者都输出 18 维动作。`--init_actor` 和 `--init_checkpoint` 仅初始化模型，不继承原任务优化器和训练计数。

后续通过验收的是独立任务 `GO2-ARX5-Box-Push-Hybrid` / `-Play`：226 维输入包括原 210 维历史和 16 维箱体/控制器状态，actor 只输出 12 维腿部动作，机械臂由 IK 控制器产生另外 6 维命令。历史中的 action 仍为实际施加的完整 18 维命令。准备阶段腿部保持默认姿态，合法接触后保持手臂参考姿态并由腿部推进，到位后松开保持状态。控制器只写关节 PD 目标，不搬动物体或机器人位姿；力矩限制仍生效。这不是原 224->18 的端到端 actor，也不能直接接入旧 210 维 MuJoCo runtime。

原生CLIMB训练使用地形网格中的固定支撑面。MuJoCo最终组合则持续使用同一个自由可移动箱体，完成推动、登箱和上高台；固定箱原生结果与这一组合结果分别报告。

## 3. 判定与已修正问题

PUSH 完成需要有效末端接触历史、箱体距离目标小于 0.12 m、速度小于 0.08 m/s，并稳定 0.5 s。身体/腿/近端机械臂碰箱、机器人跌倒或箱体倾覆是失败。USD 将手腕和指部的多个碰撞形状合并在 `link6` 刚体下，`end_effector` 是无碰撞坐标标记；当前要求过滤力反向水平分量超过 1 N 且占合力 80% 以上、平均接触点在前推面内部且距末端小于 0.07 m。手部受力大于 1 N 却未通过这些判据时，由 `invalid_hand_contact` 终止，不再只取消接触奖励。混合任务另有低姿态终止与物理失败惩罚。接触目前按控制步检查，平均接触点也不是逐碰撞形状识别，不能据此宣称逐物理子步接触完全覆盖。

CLIMB 完成需要四足落在箱顶安全区域并有向上接触力，脚高与真实扫描到的顶面一致，机器人距箱顶目标小于 0.20 m、线速度小于 0.15 m/s、角速度小于 0.4 rad/s、姿态直立并保持一秒。机身碰地形和明显倾倒都失败。没有为了提高数字降低停稳或碰撞标准。

训练诊断已修正：

- 显式执行器的 `joint_effort_limits=1e9` 是求解器默认值，不能证明内部 PD 无限力矩；新任务分别明确设置并打印内部和求解器限制。实际评估的最大力矩/限幅比为 1.0，但瞬时关节速度仍可能超过求解器设定值，未宣称速度严格逐步满足硬件限制。
- 原版奖励没有有效完成事件奖励，且旧 PUSH 只奖励向前速度；现在成功奖励按控制周期抵消积分缩放，推进收益必须来自有效接触下朝目标移动，越过目标继续向前不会得正分。
- 旧 NAV 策略的末端位置命令训练在机械臂 `link0` 安装座坐标系；初版 PUSH 错用 root 坐标系。当前统一为 link0，使用实际箱面目标并移除无关的随机末端姿态奖励。几何与真实 Isaac 合约检查均通过。
- 初版确定性评估在训练 runner 创建阶段阻塞；改为直接恢复已知未归一化 ELU actor，经输出等价测试验证，不在评估中构造优化器。
- 原评估按时间收集数量不等的回合，下面的早期数字仅作诊断；当前 `--episodes-per-env` 固定各起点权重，并单列未完成回合。

## 4. 已完成训练与评估

所有权重位于服务器仓库的 `logs/rsl_rl/`，未替换已发布部署策略。

| 训练 | 权重路径（相对 logs/rsl_rl） | 状态 |
| --- | --- | --- |
| CLIMB 第一轮，1000 更新 | `go2_arx5_box_climb/2026-09-12_19-27-10_normal_limits/model_999.pt` | 完成约 4915 万步；课程未到目标高度 |
| PUSH 第一轮，600 更新 | `go2_arx5_box_push/2026-09-12_20-16-32_normal_friction/model_599.pt` | 完成约 1475 万步；奖励/末端坐标问题尚未修正 |
| CLIMB 第二轮，1200 更新 | `go2_arx5_box_climb/2026-09-12_22-01-10_completion/model_1199.pt` | 完成约 5898 万步；训练课程均值 0.584，尚未达标 |
| PUSH 第二轮，1000 更新 | `go2_arx5_box_push/2026-09-12_22-03-12_completion/model_999.pt` | 完成约 4915 万步；有完成奖励，仍使用旧错误末端坐标 |

早期独立诊断：

- CLIMB 第一轮：7.5 cm 箱 78/112 次完成成功；25 cm 箱 0/481，多为机身接触或倾倒。
- CLIMB 第二轮：10 cm 箱 54/183 次成功；20 cm 箱 0/680，其中 673 次机身接触终止、7 次倾倒。
- PUSH 第一轮：5 kg/摩擦 0.4 为 0/64，37 次超时、27 次非末端碰箱。

这些是未限定每起点回合数的诊断记录，不能作为正式 32 起点达标率。结构化文件在 `/mnt/yuanyue/data/box-skills-eval/`，日志在 `/mnt/yuanyue/logs/skill-retraining/`。

第三轮：CLIMB 使用 `clearance_010_020`，集中于 10-20 cm 并加入支撑高度净空反馈；PUSH 使用 `corrected_mount`，从原 NAV 权重在正确 link0 坐标系下重新初始化。CLIMB 在录像和轨迹中确认退化为低姿态静止，已于约第 649 次更新定向中断，保留到 model_600.pt 的检查点，没有完成原定 800 次更新。PUSH 完成 800 次更新，权重为 `go2_arx5_box_push/2026-09-13_16-30-02_corrected_mount/model_799.pt`，耗时约 2757 秒。

该 PUSH 权重在 5 kg/摩擦 0.4 的 32 固定起点中，12 秒预算为 0/32（30 超时、2 非末端接触）。保留相同物理参数与到位标准、仅将预算显式改为 30 秒的诊断为 2/32（27 超时、2 非末端接触、1 倾倒）。因此问题不只是时间预算，还包括持续接触和推进；该策略未达标，未部署。两种预算的数据分别为 `push-mount799-normal.json` 和 `push-mount799-30s.json`，不能混报成功率。

### 趴地问题专项修正

用户观看录像后指出机器人落地即趴住。原片为 `climb-clearance.mp4`，第三轮 model_200.pt、15 cm 固定箱；轨迹确认约一秒后 base 高度降至 0.11 m，之后不再前进。该状态没有触发原机身接触或倾倒条件，且静止仍获得速度跟踪和目标接近的正收益。

独立 [check_box_startup.py](../scripts/check_box_startup.py) 用默认姿态 PD 控制进行四秒检查：将初始 base 从 0.55 m 改到 0.33 m 后，16/16 环境均站稳，最后四足支撑；一秒后的最低 base 高度范围约 0.253-0.277 m。因此不能把该视频归因于物理模型无法站立。

当前仅对新 Box-Climb 任务采取以下修正：

- 初始高度为 0.33 m，随机关节初态下脚离地约 1-6 cm，减少原约 24 cm 的自由落体。
- 增加支撑面相对的 `low_posture`：初始化宽限 0.4 s 后，机身净空持续低于 0.18 m 达 0.2 s 则失败，并扣除失败奖励；它不是完整站起恢复控制器。
- 速度跟踪改为相对静止的收益差，接近奖励使用向目标的移动速度；零速度不再从这两项获得正收益。
- 净空反馈改为较宽的连续误差反馈，避免严重低姿态时原窄指数核几乎没有信号；成功判定继续屏蔽所有失败项。

六项纯奖励合约和真实对照通过：新增约束下默认 PD 仍 16/16 站稳；原趴地检查点 16/16 都失败，15 次 low_posture、1 次机身接触，在最多 41 控制步（约 0.82 s）内结束。它是错误行为被检出，不是策略已经学会站立。

恢复训练前，以原始 CLIMB model_1499.pt 验证 7.5 cm 箱的 16 个固定起点：11 成功、2 超时、3 机身接触，没有 low_posture。随后从该未退化权重完成 `standing_guard` 的 600 次更新，约 2949 万环境步、1395 秒，不再从趴地检查点继续。最终权重为 `go2_arx5_box_climb/2026-09-13_17-02-51_standing_guard/model_599.pt`，SHA-256 为 `15a7ea74b870bdfe7e9b4724b31a8ae8bae2f5e84267945e5c96dd486720d739`。

最终关闭探索噪声、seed=101、32 固定起点、每起点一回合、12 秒预算的结果：

| 箱高 | 成功 | 其他终止原因 | 结论 |
| --- | --- | --- | --- |
| 20 cm | 30/32 | 2 超时，无机身接触/倾倒/低姿态失败 | 通过本次名义固定箱检查 |
| 25 cm | 26/32 | 4 超时、1 机身接触、1 倾倒 | 未达到 29/32 门槛 |
| 30 cm | 0/32 | 7 超时、24 机身接触、1 低姿态 | 未达标 |

统计来自 `climb-standing599-0.20.json`、`climb-standing599-0.25.json` 和 `climb-standing599-0.30.json`。这里只验证了当前固定箱面和种子分布，未覆盖可移动箱支撑、MuJoCo 部署、更多尺寸和最终高平台组合。

另录制 seed=101 的单环境 20 cm 箱试验：252 控制步、约 5.04 秒完成，四足停稳一秒，无机身接触、倾倒或低姿态失败。录像已保存在本地 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-13/climb-20cm-standing599.mp4`，1280x720、25 FPS、126 帧，视频 SHA-256 为 `6f78debdf6823f6a3f7a0b9d8853791bca2193a27af27d573f4d6e6be2944027`。对应 JSON 和预览在同目录；原失败片段保留，不以成功录像替代批量统计。

专项数据：`/mnt/yuanyue/data/box-skills-eval/startup-033-guarded.json`、`climb-prone-guarded.json`、`climb-original-guarded.json`。本地原录像在 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-13/climb-15cm-iter200.mp4`；对照录像为同目录 `climb-standing-reference-075cm.mp4`。后者展示站立和前进，但这个单回合仍因未满足连续四足停稳而超时，不能当作成功样本。

### 正常摩擦混合 PUSH

`contact_stage030` 的 224->18 训练完成 400 更新后，32 固定起点的 30 cm 目标仅 1/32、60 cm 目标 0/32，未达标。其后启用上述混合控制方案，保持 5 kg、静/动摩擦均为 0.4、恢复系数 0、刚体线性/角阻尼 0。地面摩擦为 1.0、组合方式为 min，未恢复旧 0.005 低阻箱参数。

| 训练 | 权重路径（相对 logs/rsl_rl） | 独立结果 |
| --- | --- | --- |
| 混合 30 cm，500 更新 | `go2_arx5_box_push_hybrid/2026-09-13_23-43-34_nominal030/model_499.pt` | seed=101，30 cm 30/32；60 cm 27/32，未通过 60 cm 门槛 |
| 混合 60 cm，400 更新 | `go2_arx5_box_push_hybrid/2026-09-14_01-09-58_nominal060_guarded/model_399.pt` | seed=101、102 各 30/32；加入严格手部终止后 seed=103 仍 30/32 |

第二轮共 19,660,800 环境步、1580.68 秒，从前一模型初始化，探索标准差 0.15，学习率 0.0001，加入低姿态/物理失败惩罚。检查点 SHA-256：`65ccf270000f57af0e3842b25e9d563fb17a0ca8da0af9f6e6f91cd573255cae`。验收均为每环境一回合、12 秒、关闭探索噪声，箱尺寸 1.2 x 1.2 x 0.25 m、目标平移 0.60 m。严格 seed=103 的另外两回合分别为近端/身体碰箱与无效手部接触；没有超时、倾倒、箱体倾覆或低姿态失败。

严格检查文件为 `/mnt/yuanyue/data/box-skills-eval/push-hybrid060-399-strict.json`。此前三组评估及手部接触审查文件分别带 `seed101`、`seed102` 和 `contact-audit` 后缀；最后一组的 30 个成功回合没有未分类手部受力。这个结果只覆盖名义质量/摩擦和当前小范围起点扰动，尚未证明 3-8 kg、摩擦 0.2-0.6 全范围能力。

单环境 seed=101 录像实际前移 0.631 m、最终目标误差 0.050 m、4.58 秒完成，无非法接触和姿态失败。侧视完整录像为服务器数据目录下 `push-hybrid060-399-strict.mp4`。新评估 schema v3 在自动 reset 前记录每个回合的真实箱体位置、目标误差、速度与终止标志；不再用末次稀疏轨迹点替代终态。异常栈在关闭 Isaac 前打印，评估失败返回非零退出码。

导出文件 SHA-256 为 `8556fd3196a29d46d94c27ce8cee014a42025fd879d3e0e6f87ca2e7dce22074`，129 组 CPU 输入的导出/重载对照误差为 0；完整物理录像轨迹的 229 步 actor 对照误差也为 0。`check_push_hybrid.py` 在严格配置下通过 226/12/18 接口、4/4 接触准备和局部 reset 隔离检查。模型包位于本地 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14/`，manifest 明确 MuJoCo 物理部署尚未通过；早期导出包中的评估先于严格手部终止，不应混用其控制源码 hash。

### 20 cm 箱高兼容

同一 60 cm 策略直接用于 20 cm 高箱体，原到位统计为 32/32，但 8 个回合出现短暂手部受力未通过指尖判据。加入 `invalid_hand_contact` 后严格复测为 24/32，另外 8 次均为该项失败。故不能把原始 32/32 当作最终推箱与登箱组合的前置验收。

20 cm 专项训练 `go2_arx5_box_push_hybrid/2026-09-14_01-53-59_height020_strict` 已完成 300 更新、1221.42 秒，seed=8、2048 环境。最终 model_299.pt 的训练成功比例为 0.9983，但独立 seed=101/102 仅 28/32、30/32，均有无效手部接触失败，不满足每组至少 29/32 的门槛。

终态审计发现两类问题：固定关节接触保持时，末端可能随机身运动抬到箱体上沿；原停车条件使用目标圆心距离，横向偏移会使箱体错过停车区并继续向前。后一问题已修正为沿任务世界 +X 推进方向的剩余距离，同时用于速度命令、手部撤离和 IK 释放。完成条件仍为原二维目标误差小于 0.12 m、速度小于 0.08 m/s、停稳 0.5 s，未放宽。model_299.pt 经停车修正后不再越推越远，但仍为 28/32、29/32。

同轮中间权重在 seed=101 验证集上的比较为 model_100.pt 30/32、model_200.pt 31/32，据此选定 model_200.pt，再使用未参与选择的 seed=103/104 验收：

| 评估种子 | 成功/完成 | 失败 | 成功回合最大终点误差 | 成功回合最大终止箱速 |
| --- | --- | --- | --- | --- |
| 103 | 31/32 | 1 次 invalid_hand_contact | 0.08295 m | 0.0000780 m/s |
| 104 | 31/32 | 1 次 invalid_hand_contact | 0.07901 m | 0.0000863 m/s |

两组均无未完成、超时、机身接触、倾倒、箱体倾覆或低姿态失败。箱体为 1.2 x 1.2 x 0.20 m、5 kg，静/动摩擦均为 0.4、刚体阻尼 0，目标平移 0.60 m。该结果不外推到其他尺寸、质量、摩擦或大范围起点。

- 选定检查点：`go2_arx5_box_push_hybrid/2026-09-14_01-53-59_height020_strict/model_200.pt`；SHA-256 `431bbd8bf37992e26176fc7854712ec7ff65c1c0b25289e2420dab6e7814fd53`。
- 导出包：`/mnt/yuanyue/data/box-skills-eval/push-height020-stop200-bundle/`；policy SHA-256 `9668a27832ec34764a9f0a69ddc6b36b70fc133395403ca6fcdce91466773b82`。包内保存参数、两份独立报告和当前控制源码 hash；部署必须包含本次有向停车修正，不能只复用训练时的旧代码。
- 本地录像：`/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14/push-height020-stop200.mp4`，同目录有 `-video.json`、`-preview.jpg` 和导出包。单环境 seed=103 在 5.96 秒完成，实际位移 0.51986 m、目标误差 0.08101 m、终止箱速约 0.0000302 m/s；这是带 12 cm 容差的 60 cm 目标，不是精确移动 60 cm。149 帧、25 FPS，已解码检查；所有 298 个真实观察的导出输出对照误差为 0。
- 评估 schema v4 在 reset 前保存手部力、接触点、末端位置和箱体姿态，并将终止控制步计入异常手接触统计。旧 v3 的异常步数可能为零而终止原因非零，不能据旧计数推断无异常。

已接受的 CLIMB actor 保持冻结。一次接触切向 IK 补偿实验在固定验证集上 0/32，已撤回；当前选定模型沿用原接触保持方式及严格判据。

### MuJoCo 候选与连续支撑诊断

新增 [box_push_runtime.py](../mujoco/reconfigurable_navigation/box_push_runtime.py) 与 [check_box_push_policy.py](../mujoco/check_box_push_policy.py)：226 维输入、12 维腿部动作、18 维组合动作历史，复用训练侧的限速和接触规则，使用 MuJoCo Jacobian 与零速度重力项做 IK/PD 补偿。模型包含真实自由箱、全机械臂碰撞和力矩限制，不写箱体运动轨迹。当前固定执行器延迟 profile 和被动关节摩擦只是候选配置，未证明两引擎完整动力学等价。

本地 MuJoCo 3.12.0：默认姿态下的机械臂准备检查通过；完整 PUSH 的 seed=0 通过，32 起点为 25 成功、6 次无效手接触、1 次其他部位碰箱，未达到 29/32 门槛。报告 `push-mujoco-stop200-seeds32.json` 在上述本地目录。部分失败在初次接触后抬到箱体上沿；半物理步长且保持实际延迟/控制频率的对照只改善五个早期失败中的一个，不能归结为单纯积分分辨率问题。导出 manifest 中的 `mujoco_physical_validation_passed` 继续为 false。

[check_box_support_sequence.py](../mujoco/check_box_support_sequence.py) 使用同一 MjModel/MjData、自由 5 kg 箱体、20 cm 箱和 40 cm 高台，只有一次物理初始化，技能切换无 reset。它是独立诊断，尚未接入生产 Oracle/executor。直接收臂曾导致碰箱与失稳；增加 IK 离面抬手、限速重力补偿收臂并延续执行器延迟缓存后，seed=0 完成 PUSH 4.82 s、离面 1.64 s、收臂 0.94 s。随后 CLIMB_BOX 在 3.40 s 因 bad_orientation 失败，最多三足支撑，未进入高台阶段。PUSH 后箱体额外位移约 1.5 mm，因此此单例不能认定是箱体明显滑走导致失败，也不证明移动箱承重已达标。完整报告为 `box-support-sequence-rate-limited.json`。

### 2026-09-14 迁移根因修复

匹配同一批 Isaac 物理状态后，确认旧 MuJoCo 候选存在三类差异：

1. 机械臂安装座低 0.01 m，机器人总质量为 18.74598 kg，而训练实际为 20.54928 kg；旧 MJCF 的足部、机身及部分连杆惯量不同，并缺少训练 USD 中 `end_effector` 无碰撞刚体的 1 kg 质量。新配置刻意复现训练资产，不代表这一末端质量已由实机测量确认。
2. 当前 Isaac `force_matrix_w` 是法向接触力，不含切向摩擦力；新 MuJoCo 分类改为相同法向定义，并保留合力供诊断。接触位置、指尖距离、推面内部和禁止非手部接触的门槛未放宽。
3. 当前 Isaac CircularBuffer 首次写入会用首个目标填满历史，旧候选却使用零目标预填。新 Box-Climb/Box-Push runtime 修正首命令行为，并提供执行器队列的跨技能继承；旧生产 CLIMB runtime 的历史兼容行为未改动。

实现入口为 [box_robot_profile.py](../mujoco/reconfigurable_navigation/box_robot_profile.py)、[box_robot_profile.json](../mujoco/deploy/box_robot_profile.json)、[box_climb_runtime.py](../mujoco/reconfigurable_navigation/box_climb_runtime.py)。配置从 [check_push_hybrid.py](../scripts/check_push_hybrid.py) 导出的真实质量、质心、惯量和固定变换生成；[check_box_transfer_contract.py](../mujoco/check_box_transfer_contract.py) 对 48 个匹配状态验证通过：所有刚体位置最大误差 7.16e-7 m、末端 Jacobian 6.70e-7、机械臂重力项 1.28e-5 N.m；编译后完整惯量矩阵最大误差 4.60e-8。

修正后的结果保存在本地 `artifacts/box-skills/2026-09-14/`：

| 检查 | 结果 | 报告 |
| --- | --- | --- |
| MuJoCo 名义 PUSH，seed 0-31 | 32/32 | `push-mujoco-native-contract-seeds32.json` |
| MuJoCo 名义 PUSH，新 seed 100-131 | 32/32 | `push-mujoco-native-contract-heldout32.json` |
| 原 CLIMB 权重，固定 20 cm 箱，seed 0-31 | 29/32，3 次姿态失败 | `climb-mujoco-profile-normal-support32.json` |

服务器同步后复验：5 项 runtime 契约通过，PUSH seed 100-131 仍 32/32，固定箱 CLIMB 仍 29/32，完整高台 seed 100 成功。服务器报告分别为 `/mnt/yuanyue/data/box-skills-eval/push-mujoco-profile-server32.json`、`climb-mujoco-profile-server32.json`、`box-to-platform-server100.json`。这是跨机器复现，不作为新增独立起点累加。

PUSH 仍为 5 kg、摩擦 0.4、真实自由箱和 60 cm 目标。CLIMB 支撑判定已统一为足端向上法向合力超过 2 N，最终仍要求四足位于安全顶面、满足目标误差和姿态/速度门槛并保持一秒。新 [check_box_runtime_contract.py](../mujoco/check_box_runtime_contract.py) 的 5 项回归覆盖首命令延迟、物理状态不变的队列继承、侧向力不能算支撑、向上支撑以及 NAV 实际施加动作历史。机器人被动关节摩擦和固定延迟 profile 仍属于当前部署假设，匹配几何/惯量不等于所有接触动力学完全等价。

### 独立高台候选与完整流程

原单箱 actor 在 Isaac 的抬高接近面上为 23/32（8 超时、1 机身接触）：接近支撑高 0.20 m、再上升 0.20 m，间隙 0.015 m，落点平台 2.0 x 1.6 m。新增地形参数的默认值为零抬高，原单箱几何检查继续通过。

保留原 `standing_guard/model_599.pt`，从它派生独立实验 `go2_arx5_box_climb/2026-09-14_16-14-10_raised_approach`，300 更新、2048 环境、seed 12、673.62 秒，训练相对台阶高度 0.15-0.20 m。最终 model_299.pt 在独立 seed 106/107 为 23/32、26/32，其他均超时，无物理失败，仍未通过单策略门槛。检查点 SHA-256 为 `ff62f807bedc1cc804e0b8ec19e64b8d0da11b508b0befb421796e6f13f105c0`。

[export_box_climb_actor.py](../scripts/export_box_climb_actor.py) 以 `--diagnostic` 导出该候选，包名 `climb-raised299-diagnostic-bundle`，policy SHA-256 为 `8faf4f79802fdca25afad9b2e2f1370061b24134fcee46b9313391c8fa3bffdf`。129 组导出对照误差为 0，manifest 明确 `native_validation_passed=false`。独立 MuJoCo 两级固定支撑小检查为 4/4，改起步支撑摩擦为 0.4 仍 4/4；先默认 PD 站稳一秒的对照为 3/4（1 超时）。这些小检查不作为正式独立策略达标结论。

连续流程保留真实自由 5 kg、摩擦 0.4 的 20 cm 箱体，按高台位置设置 PUSH 控制目标，并显式保存放箱目标、控制目标和实际间隙。当前控制目标在理论贴台箱心前方 0.02 m，用于补偿停车提前量，不搬动箱体。流程依次完成：PUSH、IK 离面、NAV 地面对位、收臂、登箱/箱顶定位、NAV 对准第二级入口、再次收臂、登高台/顶面定位。

NAV 地面对位采用显式 8 s 预算，必要时后退命令不小于 0.25 m/s，接近目标和朝向要求保持。箱顶 NAV 接管使用 0.5 s 目标混合，并记录实际施加动作。只有四足几何位置都在箱顶安全区域、至少三足承重、质心位于承重足支撑多边形内且距边缘超过 5 mm、姿态和速度稳定 0.3 s，才允许箱顶定位接管；最终完成条件仍要求四足承重一秒。准备 CLIMB 时可清空策略上一动作，但始终保留物理状态和执行器队列。

使用原 actor 登箱、独立候选登高台，未参与组合修正的 seed 100-131 完整任务为 **18/32**：第一段登箱 6 次机身接触、2 次姿态失败；第二段高台 3 次机身接触、3 次姿态失败。32 回合均只有一次物理初始化、技能切换零物理 reset，PUSH 均完成。报告为 `box-to-platform-heldout32.json`，含完整 runtime/资产指纹。它仍是独立诊断入口，不是生产 Oracle 集成，也不是已通过可靠性验收的组合技能。

成功完整录像为本地 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14/box-to-platform-success-seed1.mp4`，对应 JSON 和预览在同目录。约 29.72 s、743 帧、25 FPS，实际到达 40 cm 高台并四足停稳；最终 base 高约 0.679 m，PUSH 后箱体额外位移约 0.054 mm。录像对应当时四足箱顶交接版本；后续支撑多边形版本及其独立统计另外记录，不能混淆来源。

当前支撑多边形版本另录制独立评估中的成功 seed 100：`box-to-platform-success-seed100.mp4`，28.44 s、711 帧、25 FPS，配套同名 JSON 和 `-preview.jpg`。已解码检查推箱、箱顶与高台终态；最终 base 高约 0.67854 m。`mujoco-deployment-validation.json` 汇总服务器 PUSH 32/32、固定箱 CLIMB 29/32 和本地完整流程 18/32 的报告 hash、runtime 指纹及录像 hash，明确完整流程不具备生产就绪状态。原 PUSH 导出包的历史 MuJoCo 标记不回写，以此独立部署验收索引补充后续验证。

### 连续起步与完整场景专训

准备阶段新增四足支撑、腿部最大角速度小于 0.25 rad/s、机身线速度小于 0.05 m/s、角速度小于 0.10 rad/s，并连续保持 0.3 s；机械臂位置误差门槛仍为 0.10 rad。仅加强准备时，原策略在开发 seed100-131 仍为 18/32，因此没有把这项检查本身报告为成功率提升。

新增 [climb_start_states.py](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/climb_start_states.py)，只在独立 Isaac 训练回合 reset 时按可选 `prepared_start_path/phase` 导入实际站立状态，默认关闭。它保留关节观察零点和顺序，不在 MuJoCo 技能边界搬动机器人或箱体。源数据划分：

| 用途 | MuJoCo 起点 | 有效地面/箱顶状态 | 文件 |
| --- | --- | --- | --- |
| 训练 | seed16-31 | 16 / 13 | `climb-prepared-start-training16.json` |
| 原生评估 | seed200-215 | 15 / 11 | `climb-prepared-start-validation16.json` |

这不是 32 个独立场景族；原生 32 回合从有限状态池抽样，并叠加任务自身随机化。完整 MuJoCo 验收另用新起点，并把前置 PUSH 失败计入总分母。训练文件 SHA-256 为 `5af1982c5402b8dad97c9f8d3088b6c25660e1acec0a642d271f288f173afc59`，评估文件为 `3d6352d4423d6bfe9c7b148c0db530e2bf3e84983dcf4eb5ac9acb44f6df6427`；两阶段各 16/16 原生站立检查通过，关节零点未改变。导出包同时保存这两个数据文件及其 hash。

冻结的 standing599 在普通单箱、站立初态的 Isaac 对照为 32/32；同时加入摩擦 0.4 和箱后 40 cm 高台后为 0/32（31 超时、1 姿态失败）。新 `following_platform_height` 等参数复现后续高台，默认关闭；`supported_feet` 只使用当前目标箱 XY 范围内的射线计算目标高度，避免后方高台被误当作箱顶。该目标面判定、状态映射及局部 reset 隔离已由 [check_climb_start_states.py](../scripts/check_climb_start_states.py) 的 6 项回归覆盖。

两份候选均保持 253 输入、18 动作，不替换原接受权重：

| 候选 | 训练 run 与最终权重 | 训练 | 原生独立状态池评估 |
| --- | --- | --- | --- |
| 第一阶段完整场景 | `2026-09-14_19-08-05_prepared_ground_context/model_499.pt` | 500 更新，1185.16 s，seed20 | seed202 为 32/32，seed203 为 31/32，余下一次超时 |
| 第二阶段站立起步 | `2026-09-14_19-00-56_prepared_platform/model_499.pt` | 500 更新，1186.94 s，seed21 | seed202 为 26/32（5 超时、1 机身接触），未过单策略门槛 |

两者使用 2048 环境、摩擦 0.4、学习率 0.0001、初始标准差 0.12。第一阶段有后续高台几何；第二阶段为 20 cm 支撑面、再上升 20 cm、1.5 cm 间隙。包分别为 `climb-prepared-ground499-bundle` 与 `climb-prepared-platform499-bundle`，位于本地 2026-09-14 产物目录及服务器评估目录。第一阶段 checkpoint SHA-256 `f082d1ec7ad7d5a9fb79de1700b95c91552da446ff8ca7fe43bd107a91fc5f30`，policy `4cd5dafca6291234bce5b9fcdd79efc75e3c53657a779d2b554df5a1ea09feb7`；第二阶段 checkpoint `d06ca7b2839735c4aa4155c0f963437cee0eae66a9f56fccbbd633ec7085b79e`，policy `31f1268bbdf371cb9e547e0c99a9eb18ecc828941b1809d439ed59bc05ec4bd0`。两次导出各 129 组输入误差为零；第二阶段仍以 diagnostic 包导出。

控制对照与实际箱面修正：

- 同一 seed300-331，原策略组合 20/32，新双候选 27/32；当时对位仍沿世界 X 轴。
- 把前进命令从 0.30 降到 0.25 m/s，开发 seed100-115 两者均 14/16；增加限幅朝向反馈，开发 seed100-131 两者均 25/32。均无净收益，当前默认仍是 `--climb-speed 0.30 --heading-gain 0.0`。
- 推箱后箱体有偏航，而原对位点忽略旋转。`surface_entry` 现沿真实箱面法线取半箱长加 0.75 m 的接近位置，并让 NAV 对准目标面朝向；这不同于在 CLIMB 途中增加未获验证的朝向命令。旋转 90 度的几何回归通过。此后新状态记录转换到目标面朝向坐标；用于本轮训练的既有状态文件保持不变。
- 冻结上述修正后，未见 seed400-431 的新组合为 **28/32**，同条件原策略为 **15/32**。新组合失败为高台 2 次机身接触、1 次低姿态及 PUSH 1 次非法手部接触；所有成功进入第一段 CLIMB 的 31 回合都登箱完成。全部回合仅一次物理初始化，技能切换零物理 reset。

报告分别是 `box-sequence-surface-entry-heldout400.json` 和 `box-sequence-surface-entry-original400.json`。成功录像 `box-to-platform-prepared499-seed400.mp4` 约 27.78 s、695 帧、25 FPS，已解码检查，最终 base 高约 0.67906 m，配套 JSON 和预览同目录。本轮结果仍只覆盖当前布局和小范围起点扰动，不代表任意箱体或场景泛化。

间隙训练 `2026-09-14_19-52-50_prepared_platform_gaps` 已完成400更新、921.12 s，从高台499派生，seed22、学习率0.00005、初始标准差0.08，接近间隙通过地形难度覆盖0-7 cm。`approach_gap_range` 默认关闭，4项地形检查均通过，未修改物理接触或成功门槛。最终 model_399.pt 的 checkpoint SHA-256 为 `27cfc47d3ed82a5f9b2d9df375a8be23a9a5dd7ed7c26469f2cc0f67832a9da2`；诊断导出包 `climb-prepared-gaps399-bundle` 的 policy SHA-256 为 `e04e8e0ac94b67e051ae0183ff49db846f7502f67b2a77344674292fd54add04`，129组输出对照误差为零。

最终间隙候选原生验收：1.5 cm间隙、seed205为21/32（11超时）；6 cm间隙、seed206为22/32（6超时、2机身接触、2低姿态）。先前高台499在6 cm间隙为15/32。尽管较宽间隙改善，单独高台actor仍未满足29/32门槛，导出manifest保持diagnostic及原生未通过标记。

最终组合选择第一阶段 prepared_ground499、第二阶段 prepared_platform_gaps399，保留既有 PUSH200。原seed400-431复测为29/32，2次高台机身接触、1次PUSH非法手接触；进一步冻结后，全新seed500-531为29/32，失败510/525/528均为高台机身接触。后者32回合的PUSH与第一段登箱均成功，全部回合只有一次物理初始化、技能切换零物理reset。它是当前固定布局和小范围起点下的组合结果，不等同于高台actor独立达标或任意场景泛化。

跨机器对照发现：使用各自默认CPU路径时，本地AVX2为29/32、服务器AVX512为27/32；权重和源码hash相同，PUSH开始使用腿部actor后出现约1e-9量级位置差，随后接触动力学放大到不同结局。新增显式入口 [run_box_support_sequence.py](../mujoco/run_box_support_sequence.py)，只对子进程固定 `ATEN_CPU_CAPABILITY=avx2`、`MKL_CBWR=AVX2`、`DNNL_MAX_CPU_ISA=AVX2`，不改变系统配置、模型或物理参数。报告记录实际PyTorch/NumPy版本、CPU能力和线程数；目前验证的是两台支持AVX2的x86机器。经此入口，本地与服务器seed500-531均29/32，逐回合成功/失败及原因完全一致；没有宣称浮点轨迹在所有硬件上逐位相同。

最终报告为 `box-sequence-final-local500.json` 与 `box-sequence-final-server500.json`。最终录像 `box-to-platform-final-seed500.mp4` 约29.70 s、743帧、25 FPS，配套JSON及预览同目录，采用与上述验收相同数值配置，最终base高约0.67954 m，已真实登上40 cm高台并四足停稳。此前未固定CPU路径的录像和报告保留为历史，不覆盖或混用。6项起步/目标面回归、6项运行契约、4项地形检查以及既有3项固定回合评估检查通过。

同目录 `continuous-climb-validation-v2.json` 是技能收尾阶段的最终索引，绑定三份选用策略、各包manifest、两台机器的报告和录像hash，并记录29/32成功率及3次高台机身接触。索引记录当时高台actor原生独立验收未通过、尚未接入生产executor、尚未消除全部失败。本轮executor/N1集成另见 [执行与数据记录](BOX_SUPPORT_EXECUTOR_CN.md)，不改写该历史索引；原来的 `mujoco-deployment-validation.json` 也保留为上一阶段快照。

## 5. 检查与后续

CPU 合约入口：

```bash
python scripts/check_skill_initialization.py -v
python scripts/check_box_skill_rewards.py -v
python scripts/check_box_evaluation.py -v
python scripts/check_box_push_control.py -v
python scripts/check_climb_start_states.py -v
python mujoco/check_box_runtime_contract.py -v
```

地形检查在有 trimesh 的 Isaac 环境中运行 `python scripts/check_box_terrain.py -v`。真实 PUSH 命令检查使用 Isaac 启动器执行 [check_box_push_contract.py](../scripts/check_box_push_contract.py)。局部几何、奖励和加载检查通过不代表策略成功。

20 cm 名义 PUSH 和固定箱 CLIMB 的早期 MuJoCo 门槛已通过；最终连续流程独立29/32且跨机器结局一致，剩余3次高台机身接触仍需收敛。保留原接受权重和已通过的部署契约，高台诊断候选不冒充独立达标actor。复现当前组合应使用 `MUJOCO_GL=egl python mujoco/run_box_support_sequence.py --push-policy <push-height020-stop200-bundle/policy.pt> --climb-policy <climb-prepared-ground499-bundle/policy.pt> --platform-policy <climb-prepared-gaps399-bundle/policy.pt> --seed-offset 500 --seeds 1 --output-json <新报告路径> --video-path <新录像路径>`；录像需要隔离环境中的 `imageio`、`imageio-ffmpeg`。原始check入口仍可用于数值后端对照；批量执行只要存在失败仍返回非零，必须读取逐回合报告，不能把29/32写成全回合通过。世界模型训练、数据扩容和几何候选开发继续暂缓。