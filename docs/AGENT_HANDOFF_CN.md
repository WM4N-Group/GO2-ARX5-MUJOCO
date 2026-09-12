# 会话压缩与跨服务器 Agent 交接

整理日期：2026-09-11。项目：GO2-ARX5-MUJOCO，可重构导航分支。

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

本文的代码状态基于 `848a3c4`；物理技能集成提交为 `629e666`。后续提交以实际工作树和测试为准。旧外部交接文件里“尚未提交”“下一步实现 PUSH”等早期状态已经过时，不能覆盖这里较新的事实。

## 2. 一分钟恢复上下文

| 项目项 | 当前结论 |
| --- | --- |
| 最终目标 | 利用可移动物体改变可达性，实现对象中心、技能级世界模型驱动的闭环导航 |
| 系统职责 | VLM 提 WHAT，几何模块落 WHERE，世界模型评估 WHETHER，executor 负责物理执行 |
| 已实现技能 | NAV、PUSH、CLIMB；STOP 是终态；JUMP 未实现 |
| 已通过的组合 | NAV-PUSH-NAV、NAV-CLIMB-NAV、NAV-PUSH-NAV-CLIMB-NAV |
| 当前规划方式 | MuJoCo 真值状态 + 规则式 Oracle + 栅格/A*；每次仅执行最新计划的首个技能 |
| 仍依赖人工配置 | 场景语义、部分推箱目标、平台 entry/landing portal |
| 世界模型/VLM/视觉 | 尚未实现学习世界模型、VLM proposal 和 RGB-D 推理闭环 |
| 高台任务的真实进度 | 已清除台阶入口阻挡并爬台阶；未完成把箱体推成支撑物后两次 CLIMB |
| 当前迁移状态 | 新双 RTX 4090 服务器已部署，MuJoCo 回归和两次独立 GPU 冒烟通过；DDP 未验证 |
| 下一项开发 | 已有事件标签与参数化通道扫描；继续几何推面/接近/停放候选、能力 profile 和多布局场景族评测 |

## 3. 用户已经确认的方向

- 先使 CLIMB 独立训练和 Sim-to-Sim 成功，再组合复杂任务；这两项的当前阶段验收已经完成。
- 用户看过 Isaac 跟随录像，接受 CLIMB 的跳跃/前扑式动作。现阶段以成功和落稳为主，不为追求准静态步态或观感而重新训练。
- 最终 CLIMB actor 在 Isaac 中有效；当 MuJoCo 跌倒时，应先排查 adapter、执行器和动力学语义，不能直接归因于策略无效。
- 延续 Oracle-first 路线：先可靠技能、物理数据与候选验证，再世界模型，之后 VLM 和感知；不同时引入所有不确定因素。
- 环境隔离，代码由 Git 同步，日志和大规模训练产物放持久化存储。新服务器不要直接照搬旧机器绝对路径。
- 对已实现、历史验收、当前机器实测和计划项分别陈述。用户曾认为“四技能就位”，后来已澄清 JUMP 仍是占位，不能重复早期误述。

## 4. 已完成工作的压缩时间线

### 环境与稳定 locomotion

旧开发机使用隔离的 Micromamba 环境运行 MuJoCo。其 GTX 1050 Ti 不支持 Isaac Sim 5.1，训练转到旧远程 RTX 4090。恢复上游部署契约后，NAV 达到本地 100/100、旧服务器 20/20。

NAV runtime 保持上游 210 维三帧历史观测和 18 维动作，修复内容包括 action clip、root angular velocity、机械臂中立命令及启动稳定阶段。导航控制增加最小平移速度以克服近目标死区。上游 actor 没有因此重训。

### 真实物理 PUSH 与闭环 executor

为 ARX5 各 link 加入 mesh collision hull，解决视觉穿模和机身代替机械臂推箱的问题。PUSH 使用 `ALIGN -> CONTACT -> PUSH -> VERIFY -> RETREAT` 状态机，要求指尖接触，拒绝机身/腿部接触。

当前箱体是质量 5 kg、摩擦系数 0.005 的脚轮箱，不代表普通高摩擦 5 kg 箱体也已可推。真实 NAV-PUSH-NAV 本地 20/20、旧服务器 5/5，最大穿透约 3.9-7.0 mm，低于当时 10 mm 验收上限。

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

| 用途 | 已验证历史环境 |
| --- | --- |
| 旧本地 MuJoCo | Python 3.11.16，MuJoCo 3.12.0，PyTorch 2.7.0+cpu，NumPy 2.4.6 |
| 旧单卡 Isaac | Ubuntu 22.04.4，驱动 580.82.07，RTX 4090，Python 3.11.16，PyTorch 2.7.0+cu128 |
| Isaac 栈 | Isaac Sim 5.1.0.0，Isaac Lab 0.54.4，RSL-RL 5.0.1，NumPy 1.26.0 |
| Isaac Lab 固定提交 | `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`，2026-09-11 只读查询时工作树干净 |
| 新双 4090 服务器 | MuJoCo 安装及回归通过，两张 GPU 分别完成 16 环境单次 PPO；双卡并发和 DDP 未验证 |

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
- 旧 Isaac 环境实际 `pip check` 有 wheel/packaging、fastapi/starlette、isaacsim-kernel/psutil、typing_extensions 四项冲突。训练已通过不等于依赖元数据无冲突。
- 正式 CLIMB 训练已结束，但历史“GPU 空闲”不是当前事实；启动任何新作业前重新检查进程、GPU 和磁盘。

## 7. 模型与训练产物

运行 MuJoCo 必需的两个最终 actor 都已纳入代码基线：

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
- 最近一次文档上传使用单次 `git -c http.version=HTTP/1.1 push ...` 成功，未修改全局网络设置。上传后用远端 ref/hash 核对，而不只看本地 ahead/behind 缓存。
- 自动终端和用户 VS Code 的 Git 身份/认证上下文可能不同。不要伪造提交作者；需要凭据时由用户在安全入口处理。
- 多行命令粘贴日志曾出现反斜杠后的零宽字符，并返回 127。优先给单行命令并检查不可见字符，先排除 shell 粘贴问题，不据此判断策略或环境损坏。
- 旧默认终端没有 `rg`，可以使用编辑器搜索或 grep，不必为一次查询改系统环境。
- 用户报告过同一校园网、同一 VPN 在笔记本和台式机上延迟差异大。具体原因没有确诊，不能写成已解决；节点、代理模式、终端是否走代理和本地链路都是待验证因素。
- 无头 MuJoCo 回归不要求 Viewer；交互 Viewer/pynput 需要相应图形会话，不能因为 SSH 无 DISPLAY 就认定无头训练不可用。

LiveAgent 曾作为可选远程操作工具配置：旧 Ubuntu 24.04 上 AppImage 因库混用黑屏，用户级 DEB 提取版本可用；手机 WebUI/网关扩展工作暂停。它不是项目仿真或训练依赖，迁移主线无需先恢复 LiveAgent、浏览器网关或手机访问。不要把其配置库、SSH 私钥和令牌复制到仓库；历史聊天中出现的凭据应按泄露处理并轮换，不在本文重现。

## 9. 接下来做什么，以及暂时不做什么

详细范围与门槛以 [后续开发规划](RECONFIGURABLE_NAVIGATION_NEXT_PLAN_CN.md) 为准。该规划已经写出，但其中新模块尚未实现。

1. **N0**：在新机复现现有 golden regression，冻结环境、actor、控制和场景版本。
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

按迁移指南激活正确的 MuJoCo 环境后，复用现有回归，不另写一套验收入口：

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

最近一次已经核对成功的远端提交为 `848a3c46c72a5795d9d56d58c6a1a0bc8bb28b07`，它包含双卡迁移指南；前一个 `629e666` 包含物理技能集成。

本文生成时，后续开发规划、当前交接文档和根目录 AGENTS 入口属于新增本地文档，README 也有新增索引。只克隆旧的 `848a3c4` 不会获得这些新文件；迁移前需要将它们一起提交/推送，或用安全文件传输带到新服务器。后续发布状态通过实际 Git 历史和文件存在性判断，不把这段生成时状态当成永久结论。

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