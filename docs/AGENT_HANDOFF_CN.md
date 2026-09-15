# 会话压缩与跨服务器 Agent 交接

更新日期：2026-09-15。本文是当前交接入口，历史指标和安装细节分别保留在专门文档中；其他交接文件引用本文的现行状态。

## 并行小组与当前分支

2026-09-15 用户将项目分为两个小组：另一组继续 GO2-ARX5 仿真与世界模型，本工作副本负责 GO2-PIPER 技能迁移及实机落地。PIPER 独立分支 `feature/go2-piper-sim2real` 已从 `324b7a7` 建立，新增工作尚未提交或推送。

PIPER 组已完成 NAV 复用及 PUSH、CLIMB 的训练与原生验收：NAV 平地目标控制 MuJoCo 98/100；PUSH 原生两组 31/32、30/32；CLIMB 原生 30/32、32/32。箱体技能的最终 MuJoCo 检查仍为 PUSH 21/32、CLIMB 0/32，动态迁移未过，尚未实机放行。三份策略包、录像、失败报告和下一步见 [PIPER 专属交接](GO2_PIPER_HANDOFF_CN.md)，阶段目标见 [PIPER 推进计划](GO2_PIPER_SIM2REAL_PLAN_CN.md)。下方“当前快照”及历史记录描述 ARX5 基线，不是 PIPER 的验收结果，也不覆盖用户最新的小组分工。

## 当前快照

代码发布基线：`008a7958cf542fe05a318290d4534358e0704f2b`，分支 `feature/reconfigurable-navigation-oracle`，2026-09-15 已推送并核对 GitHub 远端 hash。提交包含44个代码、机器人配置和文档文件，发布前40项聚焦检查及暂存Python语法检查通过。本次后续文档整理不自动代表另一次 Git 发布，应检查实际工作树和历史。

当前完整流程在未参与调整的seed500-531上，本地和服务器均为29/32（90.6%），逐回合结局及原因一致；失败510/525/528均为高台机身接触，PUSH和第一段登箱均32/32。工况是自由5 kg箱体、摩擦0.4、20 cm箱高和40 cm高台，包含NAV对位与顶面定位，只有一次物理初始化、技能切换零物理reset。它是独立验证入口，尚未接入生产Oracle/executor，不是任意布局或实机泛化结论。详见 [箱体训练记录](BOX_SKILL_TRAINING_CN.md)。

当前选用的三个导出包为 `push-height020-stop200-bundle`、`climb-prepared-ground499-bundle`、`climb-prepared-gaps399-bundle`。它们及状态数据、录像未纳入Git，克隆源码后还需单独取得产物并核对包内manifest。

必须通过 `mujoco/run_box_support_sequence.py` 复现当前跨机器结果：它对子进程统一AVX2/单线程数值路径，并在报告记录后端。直接使用旧check入口的默认CPU路径，本地AVX2为29/32、服务器AVX512为27/32，微小推理差异会经接触动力学放大。当前显式入口已在这两台支持AVX2的x86机器验证，不改系统配置或旧生产runtime。最终本地报告 `box-sequence-final-local500.json`、服务器报告 `box-sequence-final-server500.json` 在产物目录。

最终产物根目录：本地 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14/`，服务器 `/mnt/yuanyue/data/box-skills-eval/`。索引 `continuous-climb-validation-v2.json` 绑定三份策略、源码、数值配置和报告；最终录像是 `box-to-platform-final-seed500.mp4`（约29.70s、743帧、25 FPS），同名JSON及 `box-to-platform-final-seed500-preview.jpg` 在同目录。本地/服务器报告分别为 `box-sequence-final-local500.json`、`box-sequence-final-server500.json`。不要将早期18/32索引或seed100录像当作最新版本。

2026-09-15 只读核对：本地代码HEAD为008a795，文档编辑开始前只有两个故意未跟踪的中间权重；服务器Git HEAD仍为 `6c71d1a61dbf433022b4c06f0561344884225532`，存在累计同步和资产改动。服务器实际运行源码不能仅凭HEAD判断，也不能声称整个工作树与008a795完全相同；以报告中的源码/资产/权重hash核对。不要直接pull、reset或清理该工作树。两张GPU均空闲，未发现训练或评估进程；开始新作业前仍需重查。

原生单策略状态：第一阶段prepared_ground499为32/32、31/32；最终高台gaps399为21/32（1.5cm间隙）、22/32（6cm），仍是diagnostic候选，不能用29/32组合结果冒充高台actor独立通过。高台训练400更新、921.12s已结束；当前无训练需要等待，不要重复启动旧active记录。起步数据只用于独立Isaac训练回合reset，MuJoCo技能切换不改写物理状态。训练seed16-31状态池与原生评估seed200-215状态池分离，导出包保存原始数据及hash；本轮保留原接受权重，未替换生产默认actor/Oracle。

当前下一步：收敛3次高台接触失败并完成更宽工况检查，再将新流程接入生产executor与记录接口。世界模型、VLM、RGB-D闭环和大规模数据扩容继续暂缓。原接受的CLIMB与NAV/PUSH部署基线保留；新候选与旧默认策略不得混用。

## 历史阶段记录

下面至“新 Agent 的阅读顺序”之前是2026-09-11至2026-09-14的阶段记录，包含当时尚未修复或完成的状态，仅供追溯；当前结果、下一步与运行位置以上方“当前快照”为准。

最新用户指令是接受当前 20 cm CLIMB 后“继续推进”。2026-09-14 已完成 20 cm 正常摩擦混合 PUSH 的独立 Isaac 验收：5 kg、摩擦 0.4、60 cm 目标，seed=103/104 各 31/32，每组一次无效手部接触失败，无未完成。用户需要通过移动箱体形成中间台阶登高台；已接受的 CLIMB actor 冻结。详见 [箱体技能训练记录](BOX_SKILL_TRAINING_CN.md)。暂缓世界模型训练、几何候选与大规模数据扩容。

PUSH 使用新 `GO2-ARX5-Box-Push-Hybrid`：226 维输入、12 维腿部 actor 输出，由限速 IK 合成完整 18 维关节命令，保留力矩限制。当前 20 cm 候选是 `2026-09-14_01-53-59_height020_strict/model_200.pt`，不是最后的 model_299.pt，也不是此前 25 cm 的 model_399.pt。新增有向停车避免侧偏时越过目标继续推进；必须部署匹配的控制代码。新 policy SHA-256 为 `9668a27832ec34764a9f0a69ddc6b36b70fc133395403ca6fcdce91466773b82`，导出包 `push-height020-stop200-bundle` 已下载本地，随机输入和完整录像轨迹的数值对照均为 0 误差。旧部署 actor 未替换。

CLIMB 趴地问题已修复：近地面初態、low_posture 失败和静止零收益后，固定箱 20/25/30 cm 为 30/32、26/32、0/32。用户接受 20 cm 阶段，25/30 cm 提升暂不作为下一优先项。现有实验性 MuJoCo Box-Climb 检查尚未通过随机起点门槛，需排查 adapter，不能把固定箱成功当作可移动支撑箱与高台组合已完成。

2026-09-14 继续推进后，MuJoCo PUSH 已修复并通过 seed 0-31 和独立 100-131，各 32/32；原 20 cm CLIMB 在新固定箱 profile 中为 29/32。根因包括旧 MJCF 质量/惯量和 1 cm 安装偏移、将法向力误作合力、首命令延迟缓冲误用零目标。新 `mujoco/deploy/box_robot_profile.json` 复现训练实际 20.54928 kg 机器人（含 USD 的 1 kg 无碰撞末端刚体），48 匹配状态的几何/Jacobian/重力对照通过；这不是实机惯量校准。旧生产 runtime 和 actor 均未替换。

真实移动箱到 40 cm 高台的无 reset 连续流程已跑通，独立 seed 100-131 为 18/32，尚未达到可靠组合技能要求。失败为第一段 CLIMB 6 次机身接触、2 次姿态失败，第二段 3 次机身接触、3 次姿态失败。报告 `artifacts/box-skills/2026-09-14/box-to-platform-heldout32.json`；已保存完整成功录像 `box-to-platform-success-seed1.mp4`（约 29.72 s）与对应 JSON、预览。流程包含真实 NAV 对位、收臂和箱顶定位；箱顶允许经质心支撑多边形检查的稳定三足姿态交接 NAV，最终仍需四足停稳一秒。只有一次物理初始化，所有切换均无物理 reset。

高台使用独立派生候选 `go2_arx5_box_climb/2026-09-14_16-14-10_raised_approach/model_299.pt`，300 更新已完成（673.62 s）；原生 Isaac 两级固定支撑 seed106/107 为 23/32、26/32，其他均超时，未过单策略门槛。诊断导出包 `climb-raised299-diagnostic-bundle` 的 policy SHA-256 为 `8faf4f79802fdca25afad9b2e2f1370061b24134fcee46b9313391c8fa3bffdf`，两项验证标记仍为 false。不能替换已接受 standing_guard599 或宣称高台 actor 独立达标。下一项提高两段连续 CLIMB 的可靠性；当前训练已结束，不要按旧 active 标记重启。

此前 seed100 阶段录像为本地 `artifacts/box-skills/2026-09-14/box-to-platform-success-seed100.mp4`（28.44 s、711 帧）及配套 JSON/预览；同目录 `mujoco-deployment-validation.json` 绑定该阶段结果、代码/资产和录像 hash。服务器当时复验 PUSH32/32、固定箱 CLIMB29/32、完整seed100成功和5项runtime契约。代码/文档备份在 `/mnt/yuanyue/backups/box-transfer-complete-fj2jAc/`，新源码与配置已同步；本地/服务器 MuJoCo 环境已安装 imageio 与 imageio-ffmpeg，pip check 均通过。本段为早期阶段记录，最新结果见文首。

新录像在本地 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14/push-height020-stop200.mp4`：约 5.96 秒，实际推移 0.51986 m，距 60 cm 目标误差 0.08101 m，符合 12 cm 容差。配套 `-video.json`、预览、导出包、MuJoCo 批量失败报告及 `box-support-sequence-rate-limited.json` 均在同目录。评估 schema v4 补记终止控制步的异常接触；旧 v3 的零异常步统计可能漏掉导致终止的接触，须同时读终止原因。

本段历史整理日期：2026-09-11。项目：GO2-ARX5-MUJOCO，可重构导航分支。

部署与开发更新：yuanyue 新机已完成 MuJoCo 与逐卡 Isaac 验收；N1 已完成技能边界记录、完整起点快照、跨进程重放和 600 请求/542 条执行记录的 pilot。最新开发边界见 [N1 快照与数据记录](N1_REPLAY_DATA_CN.md)，环境路径和已知冲突见 [新服务器记录](YUANYUE_SERVER_STATUS_CN.md)，不要将下文的早期生成时状态视为当前状态。

2026-09-12 后续：在 `f857437` 基础上新增 schema v2 控制步事件、PUSH 阶段和技能失败原因；本地与 yuanyue 服务器分别通过 32 项合约，默认复杂课程 10/10、CLIMB 切换 10/10、PUSH 5/5，以及 5 个技能各三次重放。服务器 60 请求为 44 成功、9 失败、7 拒绝，另有 6 截断/3 拒绝的预算验收，详见 [过程事件记录](N1_SKILL_EVENTS_CN.md)。新版文件已备份同步，远端 Git HEAD 仍为旧基线，发布状态以实际 Git 为准。下一项为参数化场景、能力 profile 与场景族划分。

本地 SSH 别名 `go2-yuanyue` 已配置新服务器端口 46308，用户已安装公钥；强制不复用 ControlMaster 的全新 BatchMode/publickey 登录验证通过。后续使用 `ssh -o BatchMode=yes -o StrictHostKeyChecking=yes go2-yuanyue`，无需让用户反复创建密码会话。别名/私钥不随仓库迁移，新机器需单独配置。

后续 N2 首步已实现并同步：`PassageScene` 用 MjSpec 参数化真实通道，场景配置进入快照和候选来源。服务器累计 40 项合约通过，十类单变量场景 7/10 接受；高摩擦接触超时、目标变化后的 NAV 超时和超质量规则拒绝均保留原始结果。28 个起点的 84 候选为 50 成功、6 失败、28 拒绝，所有原动作重现，九个代表起点各三次重放通过。详见 [参数化通道记录](N2_PASSAGE_SCENES_CN.md)。下一项是几何推面/接近/停放候选及多布局族评测，不能把本次单布局扫描视为能力标定或泛化完成。

本文将此前多轮对话中的项目目标、已验证事实、用户决策、排错结论和未完成事项压缩为可随仓库迁移的上下文。它不是逐字聊天备份，不含密码、密钥、令牌或原始调试日志，也不会迁移某个服务的内置聊天记忆。

## 1. 新 Agent 的阅读顺序

1. 先阅读本文，恢复项目上下文。
2. 检查实际 `git status`、分支和最新提交，确认当前机器是旧开发机、旧单卡服务器还是新双卡服务器。
3. 阅读 [后续开发规划](RECONFIGURABLE_NAVIGATION_NEXT_PLAN_CN.md)，区分建议路线和已实现功能。
4. 涉及部署时阅读 [双 RTX 4090 迁移指南](SERVER_MIGRATION_2X4090_CN.md)。
5. 需要研究目标和数据规格时阅读 [原始设计与实施计划](RECONFIGURABLE_NAVIGATION_PLAN_CN.md)。

当前代码发布基线为 `008a795`；`848a3c4` 是早期迁移指南提交，`629e666` 是旧物理技能集成提交。后续状态以实际工作树和测试为准。旧外部交接中的“尚未提交”“下一步实现 PUSH”等早期安排不能覆盖当前快照。

## 2. 一分钟恢复上下文

| 项目项 | 当前结论 |
| --- | --- |
| 最终目标 | 利用可移动物体改变可达性，实现对象中心、技能级世界模型驱动的闭环导航 |
| 系统职责 | VLM 提 WHAT，几何模块落 WHERE，世界模型评估 WHETHER，executor 负责物理执行 |
| 已实现技能 | NAV、PUSH、CLIMB；STOP 是终态；JUMP 未实现 |
| 旧生产基线 | NAV-PUSH-NAV、NAV-CLIMB-NAV、清除入口后爬固定台阶；仍保留 |
| 当前规划方式 | MuJoCo 真值状态 + 规则式 Oracle + 栅格/A*；每次仅执行最新计划的首个技能 |
| 仍依赖人工配置 | 场景语义、部分推箱目标、平台 entry/landing portal |
| 世界模型/VLM/视觉 | 尚未实现学习世界模型、VLM proposal 和 RGB-D 推理闭环 |
| 高台任务的真实进度 | 独立移动箱支撑流程29/32，原生高台actor未独立达标，未接入生产executor |
| 当前迁移状态 | 双4090已部署，分卡并行训练和MuJoCo复现完成；DDP未验证 |
| 下一项开发 | 改善高台接触失败、扩大技能范围检查、准备生产集成；学习世界模型与数据扩容暂缓 |

## 3. 用户已经确认的方向

- 最新优先级是正常摩擦推箱、登移动箱及高台的真实物理可靠性；单独actor、组合流程和旧基线的成绩分别记录。
- 先使 CLIMB 独立训练和 Sim-to-Sim 成功，再组合复杂任务；这两项的当前阶段验收已经完成。
- 用户看过 Isaac 跟随录像，接受 CLIMB 的跳跃/前扑式动作。现阶段以成功和落稳为主，不为追求准静态步态或观感而重新训练。
- 最终 CLIMB actor 在 Isaac 中有效；当 MuJoCo 跌倒时，应先排查 adapter、执行器和动力学语义，不能直接归因于策略无效。
- 延续 Oracle-first 路线：先可靠技能、物理数据与候选验证，再世界模型，之后 VLM 和感知；不同时引入所有不确定因素。
- 环境隔离，代码由 Git 同步，日志和大规模训练产物放持久化存储。新服务器不要直接照搬旧机器绝对路径。
- 对已实现、历史验收、当前机器实测和计划项分别陈述。用户曾认为“四技能就位”，后来已澄清 JUMP 仍是占位，不能重复早期误述。

## 4. 历史基线时间线

本节为早期NAV、低摩擦PUSH和低台阶CLIMB的验证历史；不代表最新箱体候选，也不能覆盖当前快照中的待办。

### 环境与稳定 locomotion

旧开发机使用隔离的 Micromamba 环境运行 MuJoCo。其 GTX 1050 Ti 不支持 Isaac Sim 5.1，训练转到旧远程 RTX 4090。恢复上游部署契约后，NAV 达到本地 100/100、旧服务器 20/20。

NAV runtime 保持上游 210 维三帧历史观测和 18 维动作，修复内容包括 action clip、root angular velocity、机械臂中立命令及启动稳定阶段。导航控制增加最小平移速度以克服近目标死区。上游 actor 没有因此重训。

### 真实物理 PUSH 与闭环 executor

为 ARX5 各 link 加入 mesh collision hull，解决视觉穿模和机身代替机械臂推箱的问题。PUSH 使用 `ALIGN -> CONTACT -> PUSH -> VERIFY -> RETREAT` 状态机，要求指尖接触，拒绝机身/腿部接触。

该旧生产场景的箱体为质量5 kg、摩擦系数0.005的低阻基线，不能代替新5 kg/摩擦0.4任务的验收。旧真实NAV-PUSH-NAV本地20/20、旧服务器5/5，最大穿透约3.9-7.0 mm，低于当时10 mm验收上限。

executor 每个技能后重新观察并规划，不盲目执行原计划后缀。已验证失败预算和 Viewer 中断的区分；完整站立恢复、跌倒恢复仍未完成。

### CLIMB 训练与 Sim-to-Sim

独立任务 `GO2-ARX5-Climb` / `GO2-ARX5-Climb-Play` 已训练到 `model_1499.pt`，actor 输入 253 维，其中 187 点高度扫描，输出 18 维动作。地形为倒金字塔台阶课程，台阶高度范围 0.02-0.12 m。

最终 Isaac 评估的 16 个环境都完成 1000 步，均行进超过 3 m。最终训练 reward 约 65.25，episode length 约 979/1000，terrain level 约 4.69/7。JIT、ONNX 和录像已导出。

早期同一 actor 在 MuJoCo 立即跌倒；最终修复集中在：

- DelayedPD 逐物理步延迟，以及 reset 的零目标缓冲。
- 腿部关节速度限制及每物理步位置积分限制。
- 训练时机械臂 PD 增益，避免错误的 joint target 提前裁剪。
- 当前 runtime 使用 free-joint qvel 构造与 Isaac 一致的 root body velocity；此前的 `mj_objectVelocity` 用法产生了不一致的角速度观测。这不是对该 API 在所有场景下的否定。

最终 actor 未修改、未重训：在 MuJoCo 低台阶场景下，前进命令 0.4/0.5/0.6 m/s 均运行 20 秒，无 base-terrain contact，位移分别为 6.56/8.16/9.82 m。

### 技能切换与复杂课程

`ClimbRuntime` 从 Viewer 抽离并挂载共享的 `MjModel/MjData`。NAV 与 CLIMB 切换仅调整内部历史、last action 和延迟缓冲，不 reset 整个物理世界。

进入 CLIMB 前增加 0.5 s 默认姿态准备，避免 NAV/PUSH 遗留机械臂姿态影响启动。NAV-CLIMB-NAV 通过 10/10。生产 Oracle 加入平台 portal 和阻挡入口的箱体判断，复杂课程通过 10/10，每次通常 6 次重规划、一次物理 reset，并有指尖接触、无机身接触、无非法碰撞。

上述是既有分布的历史验收，不是新服务器或任意场景的泛化证明。连续同类 NAV 修正可能是正常重规划，不能仅凭阶段记录比预期多一条就认定失败。

## 5. 当前实现入口

| 入口 | 职责 |
| --- | --- |
| [run_box_support_sequence.py](../mujoco/run_box_support_sequence.py) | 当前独立移动箱高台流程的统一CPU启动入口 |
| [check_box_support_sequence.py](../mujoco/check_box_support_sequence.py) | 实际箱面对位、准备、两段CLIMB及逐阶段报告；不等于生产executor |
| [box_push_runtime.py](../mujoco/reconfigurable_navigation/box_push_runtime.py) | 226维输入、12腿动作、IK机械臂与18维组合动作历史 |
| [box_climb_runtime.py](../mujoco/reconfigurable_navigation/box_climb_runtime.py) | 新箱体执行器首命令、支撑和队列继承契约 |
| [box_robot_profile.json](../mujoco/deploy/box_robot_profile.json) | 对齐训练资产的安装/惯量配置，不是实机标定 |
| [BOX_SKILL_TRAINING_CN.md](BOX_SKILL_TRAINING_CN.md) | 当前候选、训练和对照证据的详细来源 |
| [oracle_planner.py](../mujoco/reconfigurable_navigation/oracle_planner.py) | 规则候选、路径/阻挡判断、PUSH/CLIMB 技能计划 |
| [representations.py](../mujoco/reconfigurable_navigation/representations.py) | 对象、能力、OracleObservation 和 SkillAction |
| [executor.py](../mujoco/reconfigurable_navigation/runtime/executor.py) | 逐技能执行、actor 切换与重规划 |
| [replanner.py](../mujoco/reconfigurable_navigation/runtime/replanner.py) | 从最新计划取下一个动作 |
| [safety.py](../mujoco/reconfigurable_navigation/runtime/safety.py) | 状态有效性和执行预算 |
| [locomotion_runtime.py](../mujoco/reconfigurable_navigation/locomotion_runtime.py) | 已验证的 NAV/PUSH 低层策略适配 |
| [climb_runtime.py](../mujoco/reconfigurable_navigation/climb_runtime.py) | 已验证的 CLIMB 观测/控制/延迟语义 |
| [skills](../mujoco/reconfigurable_navigation/skills/) | NAV、PUSH、CLIMB 生命周期 |
| [complex_course_env.py](../mujoco/reconfigurable_navigation/complex_course_env.py) | 固定复杂课程真值及人工 portal |
| [CLIMB 训练记录](CLIMB_TRAINING_CN.md) | 任务参数、原训练 run、指标、导出和诊断结果 |

旧 `SkillExecutionRecord` 保持兼容；可选 `on_transition` 提供前后观测、时间及中断语义，`on_skill_start` 支持保存起点快照。离线候选工具已能在独立物理分支执行参数扰动，但当前生产 Oracle 仍使用栅格/移除对象规则，尚未用物理候选评分或学习模型决定动作。

原方案中的 8x128x128 BEV、32x16 Object Tokens 和世界模型 heads 是后续规格，不要与低层 actor 的 210/253 维观测混淆，也不要声称这些训练编码已全部实现。

## 6. 环境与机器身份

当前路径如下；服务器Git工作树状态见当前快照，安装细节以 [服务器状态](YUANYUE_SERVER_STATUS_CN.md) 和 [迁移指南](SERVER_MIGRATION_2X4090_CN.md) 为准。

| 用途 | 当前路径 |
| --- | --- |
| 本地仓库 | `/home/yuanyue/re-nav/GO2-ARX5-MUJOCO` |
| 本地启动器 / MuJoCo环境 | `/home/yuanyue/re-nav/.tools/micromamba` / `/home/yuanyue/re-nav/.envs/go2-arx5-nav` |
| 当前SSH别名 | `go2-yuanyue`，端口46308；公钥登录已配置 |
| 服务器仓库 | `/mnt/yuanyue/GO2-ARX5-MUJOCO` |
| 服务器MuJoCo / Isaac环境 | `/mnt/yuanyue/envs/go2-mujoco` / `/mnt/yuanyue/envs/go2-isaac` |
| Isaac启动器 / IsaacLab | `/mnt/yuanyue/bin/isaac-python` / `/mnt/yuanyue/IsaacLab` |
| 日志与报告 | `/mnt/yuanyue/logs/skill-retraining/` / `/mnt/yuanyue/data/box-skills-eval/` |

| 用途 | 已验证环境 |
| --- | --- |
| 旧本地 MuJoCo | Python 3.11.16，MuJoCo 3.12.0，PyTorch 2.7.0+cpu，NumPy 2.4.6 |
| 旧单卡 Isaac | Ubuntu 22.04.4，驱动 580.82.07，RTX 4090，Python 3.11.16，PyTorch 2.7.0+cu128 |
| Isaac 栈 | Isaac Sim 5.1.0.0，Isaac Lab 0.54.4，RSL-RL 5.0.1，NumPy 1.26.0 |
| Isaac Lab 固定提交 | `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`，2026-09-11 只读查询时工作树干净 |
| 当前双4090服务器 | MuJoCo复现与分卡并行PPO训练完成；DDP未验证 |

旧路径仅用于寻找历史资料，不是新服务器默认路径：

```text
旧本地仓库：/home/yuanyue/re-nav/GO2-ARX5-MUJOCO
旧本地 Micromamba：/home/yuanyue/re-nav/.tools/micromamba
旧本地环境：/home/yuanyue/re-nav/.envs/go2-arx5-nav
旧训练 SSH 别名：go2-4090（只在已配置它的机器上有效）
旧训练仓库：/mnt/miaojigui/GO2-ARX5-MUJOCO
旧 CLIMB worktree：/mnt/miaojigui/worktrees/go2-climb
旧 Isaac Lab：/mnt/miaojigui/IsaacLab
旧启动器：/mnt/miaojigui/bin/isaac-python
```

不要覆盖旧服务器或 worktree 中的无关未提交文件。SSH 配置在对话期间被用户修改过；如需编辑，先读取当前内容，不假定旧别名/端点仍代表目标机器。本文不提供 SSH 凭据，克隆仓库也不会配置服务器登录。

重要兼容性约束：

- MuJoCo 与 Isaac 环境分离，尤其不要把 CPU PyTorch 或 NumPy 2.x 装进已验证 Isaac 环境。
- Isaac Sim 5.1 使用 Python 3.11，Linux pip 安装要求 glibc >=2.35。不要在迁移时同时升级 Isaac 主版本或改用不兼容的旧 Isaac Lab release。
- 旧启动器通过环境内 libstdc++ 预加载解决 `CXXABI_1.3.15`；新路径封装见迁移指南，不替换系统动态库。
- 当前MuJoCo环境的依赖检查已通过；Isaac环境既有FastAPI/Starlette元数据冲突仍需单独看待。旧单卡环境曾有更多冲突，不应照搬为新环境状态；训练通过不等于所有依赖元数据无冲突。
- 正式 CLIMB 训练已结束，但历史“GPU 空闲”不是当前事实；启动任何新作业前重新检查进程、GPU 和磁盘。

## 7. 模型与训练产物

当前组合选用的三个包均不随Git克隆。包根目录为本地 `/home/yuanyue/re-nav/artifacts/box-skills/2026-09-14/` 或服务器 `/mnt/yuanyue/data/box-skills-eval/`；各包包含 `policy.pt`、训练参数、评估与manifest。使用前核对hash和原生/组合验收标记，不能从目录名推断已达标。

| 角色 | 当前包 | 边界 |
| --- | --- | --- |
| PUSH | `push-height020-stop200-bundle` | 226->12加IK，不是旧210输入runtime |
| 第一段登箱 | `climb-prepared-ground499-bundle` | 原生独立状态池32/32、31/32；需连续起步准备 |
| 第二段高台 | `climb-prepared-gaps399-bundle` | 原生21/32、22/32，仍diagnostic；29/32是包含NAV的完整组合 |

旧生产基线所需的两个actor已纳入Git，不能用来冒充上述三个包：

- [NAV actor](../mujoco/deploy/policy/go2_arx5/policy.pt)，SHA-256 为 `d46a829f8cc2f85f19a030140092a206880e56f42aca6314be4e165e847de347`。
- [CLIMB actor](../mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt)，SHA-256 为 `ef88741365e85365486c30762bd792e57292eb8798b5f5aaa19af2ebb8495af5`。

虽然 .gitattributes 声明 LFS，但已核对的 GO2 策略、USD 子层和网格实际是完整 Git blob，不是指针，随普通克隆取得。此结论只覆盖已核对目录；仍应检查文件内容和 hash。

原始可续训 checkpoint 不在 Git：

```text
/mnt/miaojigui/worktrees/go2-climb/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/model_1499.pt
```

续训时同时迁移同一 run 的 `params/env.yaml`、`params/agent.yaml` 和所需日志。TorchScript actor 不能恢复 critic、优化器和完整训练进度。上游 NAV 策略也只有发布的部署 actor，没有对应可恢复训练 checkpoint。

旧开发机还保存：

```text
/home/yuanyue/re-nav/artifacts/go2-climb/iter1499/policy.onnx
/home/yuanyue/re-nav/artifacts/go2-climb/iter1499/climb-follow-camera.mp4
```

跟随视频曾验证为 960x540、50 FPS、999 帧，约 20 秒。它和完整训练日志不随 Git 克隆。两个本地中间 actor `policy_iter348.pt`、`policy_iter900.pt` 是诊断产物，曾特意不提交，不应为了清空工作区随意删除或上传。

## 8. 已知运行与同步问题

- Git 曾出现 HTTP 408、TLS 连接中断。约 1.1 MiB 的功能包并不大；网络问题不能靠重新 commit 解决，也不能根据错误日志末尾的 `Everything up-to-date` 判定成功。
- 一次 commit 成功、push 失败后，只重试 push；再次运行 `git commit && git push` 时，空提交会阻止后半段执行。
- 2026-09-15 发布008a795时，直连GitHub超时；经既有服务器代理及临时本地回环转发17890推送成功，随后已关闭转发，没有修改全局Git代理。上传后要核对远端ref/hash，不能只看本地缓存；不要假定临时转发仍存在。
- 自动终端和用户 VS Code 的 Git 身份/认证上下文可能不同。不要伪造提交作者；需要凭据时由用户在安全入口处理。
- 多行命令粘贴日志曾出现反斜杠后的零宽字符，并返回 127。优先给单行命令并检查不可见字符，先排除 shell 粘贴问题，不据此判断策略或环境损坏。
- 旧默认终端没有 `rg`，可以使用编辑器搜索或 grep，不必为一次查询改系统环境。
- 用户报告过同一校园网、同一 VPN 在笔记本和台式机上延迟差异大。具体原因没有确诊，不能写成已解决；节点、代理模式、终端是否走代理和本地链路都是待验证因素。
- 无头 MuJoCo 回归不要求 Viewer；交互 Viewer/pynput 需要相应图形会话，不能因为 SSH 无 DISPLAY 就认定无头训练不可用。

LiveAgent 曾作为可选远程操作工具配置：旧 Ubuntu 24.04 上 AppImage 因库混用黑屏，用户级 DEB 提取版本可用；手机 WebUI/网关扩展工作暂停。它不是项目仿真或训练依赖，迁移主线无需先恢复 LiveAgent、浏览器网关或手机访问。不要把其配置库、SSH 私钥和令牌复制到仓库；历史聊天中出现的凭据应按泄露处理并轮换，不在本文重现。

## 9. 接下来做什么，以及暂时不做什么

当前执行顺序以下列事项为准；[后续开发规划](RECONFIGURABLE_NAVIGATION_NEXT_PLAN_CN.md) 中N0-N7为保留的长期路线，不能覆盖当前技能优先级。

1. 固定源码、三份策略、机器人profile和数值启动器，复现最终报告；不按旧终端ID重启已结束训练。
2. 针对seed510/525/528的高台机身接触继续诊断，区分策略、接近几何和控制差异，保持原成功/失败判据。
3. 检查更宽质量、摩擦、尺寸、间隙及起点范围，单列边界失败，不根据当前单布局29/32外推能力。
4. 将验证后的新流程接入生产executor与数据记录，检查真实技能切换、快照兼容和失败恢复；不直接覆盖旧默认actor。
5. 技能范围与数据质量可靠后，再恢复几何候选、多布局、世界模型和VLM/RGB-D工作。

已完成或暂停的长期路线状态：

1. **N0**：新机既有基线复现已完成；改动涉及旧基线时按需回归，不重复安装或训练。
2. **N1**：已有 SkillTransition、技能起点快照和 542 条实际执行 pilot；控制步过程事件与原因已在本地和服务器验证，继续逐物理步/失稳判据及数据质量覆盖。当前仅一个场景族，不能用样本数代替泛化评估。
3. **N2**：单通道族参数化扫描与独立分支候选已贯通；继续多个几何候选和多布局族，假设移除对象不能代替真实物理教师。
4. **N3/N4**：先做特权状态 NAV/PUSH 世界模型和校准，再做模型驱动的搜索/CEM/闭环；低层支持 CLIMB 不等于模型已覆盖它。
5. **后续分支**：VLM proposal、RGB-D 输入替换、移动箱体支撑高台分别验收。JUMP、Gap Repair、搭桥和实机迁移不阻塞当前世界模型 MVP。

数据必须包含成败、边界和反事实，并按场景族拆分，防止同场景候选泄漏到测试集。有限搜索没找到解记为预算内未知，不直接伪造不可达负标签。快照还要保存历史观测和 CLIMB 延迟缓冲，不能只存 qpos/qvel。

双 4090 优先各训练一个小型 ensemble 成员/seed；MuJoCo rollout 主要使用 CPU。没有 profiling 证据前，不为使用两张卡而先引入 DDP。完整 world-model 数据集和网络尚未开始实现，不能根据目录设计图声称已有成果。

## 10. 接手后的最小检查

先运行只读检查，确认机器和分支。以下命令在仓库根目录执行：

```bash
git status --short --branch
git log -3 --oneline
nvidia-smi
```

先确认三个非Git策略包齐全，再按实际任务选择检查，不把下面的示例当作每次阅读交接后自动运行整套仿真或训练的要求。

当前连续流程的服务器单回合复现命令如下，结果写入新建目录；本地需改用上表的Micromamba环境和本地产物路径。该命令未启用录像，录像另加 `--video-path` 并设置 `MUJOCO_GL=egl`。

```bash
check_dir=$(mktemp -d /mnt/yuanyue/data/box-skills-eval/handoff-check-XXXXXX) && /mnt/yuanyue/envs/go2-mujoco/bin/python /mnt/yuanyue/GO2-ARX5-MUJOCO/mujoco/run_box_support_sequence.py --push-policy /mnt/yuanyue/data/box-skills-eval/push-height020-stop200-bundle/policy.pt --climb-policy /mnt/yuanyue/data/box-skills-eval/climb-prepared-ground499-bundle/policy.pt --platform-policy /mnt/yuanyue/data/box-skills-eval/climb-prepared-gaps399-bundle/policy.pt --seed-offset 500 --seeds 1 --output-json "$check_dir/result.json"
```

批量 `--seeds 32` 若有任何失败仍返回非零，需读取逐回合结果；不能把29/32称为全回合通过。当前40项发布前检查是已有证据，本次文档核对没有重跑策略验收。

改动旧生产行为时再复用其回归入口：

```bash
python mujoco/check_reconfigurable_navigation.py --seeds 100
python mujoco/evaluate_policy_navigation.py --seeds 100
python mujoco/check_push_skill.py --seeds 20
python mujoco/check_climb_skill_switching.py --seeds 10
python mujoco/check_complex_course.py --seeds 10
python mujoco/visualize_climb_policy.py --policy mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt --duration 20 --headless --no-realtime
```

需要查看当前复杂课程且已有图形会话时：

```bash
python mujoco/check_complex_course.py --seeds 1 --visualize
```

遇到失败先核对版本、模型、路径、启动和观测语义；保留现有用户改动。检查、安装、训练和发布分别以当前用户授权为准，不把读取交接文件当成启动长训练或推送 Git 的授权。

## 11. 发布状态与记忆如何迁移

最近一次核对成功的代码发布为 `008a7958cf542fe05a318290d4534358e0704f2b`（2026-09-15，44文件），覆盖箱体技能训练、部署配置、检查入口及当时的交接文档。早期迁移提交848a3c4和技能提交629e666仅用于历史定位。

代码基线、服务器运行目录和外部产物分别迁移：克隆源码不会取得新策略包、状态数据、录像或私有SSH/代理配置；服务器Git HEAD旧而工作树已文件同步，也不能直接reset或pull覆盖。本次文档整理后的新修改是否另行发布，应检查Git工作树和远端，不能把本文中的代码基线hash当作文档修改已经推送的证明。

支持仓库 AGENTS 约定的工具可以自动发现 [根目录入口](../AGENTS.md)，但是否自动载入取决于 agent 和编辑器配置。通用兜底是在新会话中明确要求读取这些文件；不依赖旧 VS Code workspaceStorage、内部 memory 工具或旧机器绝对路径。

给新 agent 的开场文字：

```text
请先读取仓库根目录 AGENTS.md 和 docs/AGENT_HANDOFF_CN.md，
再读取 docs/RECONFIGURABLE_NAVIGATION_NEXT_PLAN_CN.md；涉及新机环境时读取
docs/SERVER_MIGRATION_2X4090_CN.md。它们记录了此前会话的项目状态和决策。
请先核对当前 Git/机器/环境，区分历史验收与本机结果，用中文说明当前进度和下一步。
不要重新训练已验收的 CLIMB 来改善动作观感，也不要把规划项当成已实现功能。
```

后续每完成一个阶段，更新已验证行为、测试范围、当前代码基线、未完成事项和下一步入口即可；不要重新堆积所有终端输出。原始安装细节与完整研究规划分别维护在已有专门文档中。