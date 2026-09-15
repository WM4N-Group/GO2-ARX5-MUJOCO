# GO2-PIPER 实机组交接

更新日期：2026-09-15。当前分支：`feature/go2-piper-sim2real`，从 `324b7a7` 建立，当前新增工作尚未提交或推送。

本组负责 GO2-PIPER 技能训练及实机落地；另一组沿原 GO2-ARX5 方案推进仿真与世界模型。阅读 [仓库规则](../AGENTS.md)、[原项目交接](AGENT_HANDOFF_CN.md) 后，以本文恢复本组的实际进度；阶段目标见 [推进计划](GO2_PIPER_SIM2REAL_PLAN_CN.md)。

## 1. 当前决策

**三项技能的训练/复用与原生阶段验收已完成；PUSH、CLIMB 的 MuJoCo 动态迁移仍未通过，尚不能进行实机放行。**

- NAV 复用 [基础策略](../mujoco/deploy/policy/go2_piper/policy.pt)，没有重训；已在对齐惯量的 PIPER 模型上复用 `NavigateSkill` 完成平地随机目标到达与停稳验证。
- PUSH 选用姿态 IK 版本 `piper-push-pose599-bundle`：226 维输入、12 维腿部输出，IK 合成机械臂命令。必须使用末端向下倾角 0.8 rad、预接近距离 0.03 m、接触后锁臂的匹配控制代码。
- CLIMB 选用 `piper-climb-center200-bundle`：253 维输入、18 维输出，20 cm 固定箱登高及四足停稳；选择的是续训中的 `model_200.pt`，不是最后的 `model_599.pt`。
- 两项新技能均完成两组独立的 32 回合原生验收，每组至少 30/32；导出包随机输入及录像全轨迹推理对照均为零误差。
- 保留这些已通过原生验收的权重，下一步优先定位动态 Sim-to-Sim 差异。没有替换 ARX5 默认 actor，没有操作实机，没有提交或推送本轮代码。

## 2. 本轮实测

最终证据索引为 [piper-three-skills-index.json](../../artifacts/go2-piper/2026-09-15/piper-three-skills-index.json)。策略、报告、运行源码、资产及录像 hash 已核对。以下是不同层级的实测结果，不能合并成一个成功率。

| 后端 | 技能/工况 | 种子/环境 | 通过数 | 说明 |
| --- | --- | --- | --- | --- |
| MuJoCo | NAV，1-2 m 随机目标及随机终点朝向 | seed500-599 | 98/100 | 两次超时；到达后停稳 1 秒，12 cm / 0.15 rad 容差 |
| MuJoCo | NAV，低末端目标下前进 0.25 m/s | seed400-431 | 32/32 | 惯量/安装修正后通过；原始未修正模型的 0/32 仅为历史 |
| Isaac | 现有 NAV actor，默认/低末端命令原生参考 | seed10/11，各 16 环境 | 各 16/16 | 未修改原权重的低层参考，非世界模型或实机结果 |
| Isaac | PUSH，自由 5 kg、摩擦 0.4、20 cm 箱，推移目标 60 cm | seed206/207，各 32 环境 | 31/32、30/32 | 失败分别为 1、2 次非法末端接触；无超时 |
| Isaac | CLIMB，20 cm 固定箱，四足落稳 1 秒 | seed208/209，各 32 环境 | 30/32、32/32 | 第一组两次机身接触；无超时 |
| MuJoCo | 同一 PUSH 导出 actor，匹配姿态 IK 与 USD 碰撞 profile | seed500-531 | 21/32 | 11 次非法末端接触，未过迁移门槛 |
| MuJoCo | 同一 CLIMB 导出 actor，USD 碰撞 profile 与质心速度观测 | seed500-531 | 0/32 | 23 次机身接触、7 次姿态失败、2 次超时，未过迁移门槛 |

NAV 使用 3 秒启动准备，控制器内部到达阈值为 7 cm / 0.08 rad，为最终停止留出余量；最终验收仍为 12 cm / 0.15 rad。该结果覆盖平地目标姿态控制，不是障碍规划、生产 executor 集成或任意布局泛化。

PUSH 成功要求指定末端真实接触、箱体平移误差小于 12 cm、速度小于 0.08 m/s 并保持 0.5 秒；机身/腿部代推和非法指爪接触仍计失败。CLIMB 保留四足支撑、目标距离、姿态、线/角速度及持续停稳判据，未通过降低门槛获得成绩。

原生动作样例：

- [PUSH 录像](../../artifacts/go2-piper/2026-09-15/piper-push-pose599-native.mp4)：5.68 秒、25 FPS、1280x720；seed210 单回合成功，283 次实际观测的导出推理对照误差为零。
- [CLIMB 录像](../../artifacts/go2-piper/2026-09-15/piper-climb-center200-native.mp4)：3.52 秒、25 FPS、1280x720；seed211 单回合成功，176 次实际观测的导出推理对照误差为零。

录像是动作观察样例，不能替代上表的独立批量成绩。所有策略仍未取得实机验收。

## 3. 已定位的差异与未完成项

- 旧 MJCF 原始初态的四个膝关节位于关节限位外；新 PIPER 入口在唯一一次物理初始化时写入受限的名义姿态。后续 `runtime.reset()` 只重建控制历史，不 reset 物理状态。
- 已导出 [PIPER 惯量 profile](../mujoco/deploy/piper_robot_profile.json)：未随机化机器人总质量为 21.76500064 kg，旧 MJCF 为约 19.11895 kg；修正了安装高度和关节零位变换。它是训练资产对齐，不是实物标定。
- 16 组同状态对照已通过：质量及质心差为零，惯量最大差约 `4.6e-8`，位置约 `5.0e-7 m`，Jacobian 约 `1.3e-6`，重力约 `2.3e-5`。PhysX 浮动基座以质心为参考，比较 Jacobian/力矩时已显式换算参考点。
- 当前 IsaacLab 的 `root_lin_vel_b` 指向基座质心速度。PIPER CLIMB 已补上角速度与质心偏移的叉乘项；16 组完整观测对照最大差约 `2.7e-6`。早期探针的高度扫描缓存误差已修复，不能再当作当前扫描顺序错误。
- 已导出 [USD 碰撞 profile](../mujoco/deploy/piper_collision_profile.json)，包含 27 个原语/凸包。动态接触仍未一致，不能因为静态惯量、FK 和观测对齐通过，就宣布整个 Sim-to-Sim 完成。
- PUSH 必须使用可达的向下 0.8 rad 指爪姿态与 3 cm 预接近距离。水平指爪姿态在当前关节限位下不可达；不能把位置 IK、水平姿态 IK 和本次通过的倾角 IK 混用。
- 新箱体任务的仿真臂部限幅为 `20/20/15/7/5/5 Nm`、速度 `3 rad/s`，来源是仓库 PIPER 预留配置，尚未实物标定。GO2 腿部力矩和速度限制同时设置在执行器与 PhysX 层；原生 USD 的统一 100 Nm 臂部默认值没有用于最终箱体训练。
- NAV 保留已验证的直接 PD 控制配置；箱体技能采用 DelayedPD 对应运行时。0-4 步延迟扫描和静摩擦近似诊断未单独解决 CLIMB 动态失败，后续应比较逐物理步目标、力矩、接触及积分语义。
- 原生评估按环境配额记录首次回合，自动 reset 后的回合不会补成成功样本。未通过候选、消融和中止训练均保留，不混入正式验收。
- 尚未确认实物型号/固件、安装/负载、SDK 控制模式和传感器。实机要求的可用观测、状态估计、停止/急停与限幅必须单独落实。

## 4. 新增入口

| 文件 | 用途 |
| --- | --- |
| [check_piper_startup.py](../mujoco/check_piper_startup.py) | 模型、执行器/关节顺序、两份策略接口、初态限位和来源 hash |
| [piper_locomotion_runtime.py](../mujoco/reconfigurable_navigation/piper_locomotion_runtime.py) | PIPER 专属物理初始化，复用已验证的历史观测与 NAV 控制逻辑 |
| [check_piper_runtime_contract.py](../mujoco/check_piper_runtime_contract.py) | 名义姿态、训练关节顺序、控制器 reset、逐物理步中止、异常物理 reset 的 5 项检查 |
| [check_piper_locomotion.py](../mujoco/check_piper_locomotion.py) | 无头速度命令检查；逐物理步监测；输出失败、判据和完整网格 hash |
| [check_piper_native.py](../scripts/check_piper_native.py) | 在隔离源码下运行现有 TorchScript actor；核验实际导入路径，记录原生命令、首次回合和实际质量 |
| [PIPER 箱体任务](../source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_piper/box_env_cfg.py) | `GO2-PIPER-Box-Push-Hybrid`、`GO2-PIPER-Box-Climb` 及 Play 配置 |
| [train.py](../scripts/rsl_rl/train.py) | 支持 PIPER NAV 腿部初始化、显式 NAV 到 CLIMB 观测映射及 checkpoint 续训 |
| [evaluate_box_skills.py](../scripts/evaluate_box_skills.py) | 原生物理验收、终态原因、机器人/控制来源和部署 actor 全轨迹对照 |
| [check_piper_navigation.py](../mujoco/check_piper_navigation.py) | 复用 NavigateSkill 的随机目标到达与停止检查 |
| [piper_box_runtime.py](../mujoco/reconfigurable_navigation/piper_box_runtime.py) | PIPER 模型、PD、命名、末端和独立箱体场景 |
| [check_piper_box_skills.py](../mujoco/check_piper_box_skills.py) | 两类导出技能的 MuJoCo 检查，明确选择姿态 IK 与碰撞 profile |
| [check_piper_native_alignment.py](../mujoco/check_piper_native_alignment.py) | 同状态惯量、质心、FK、Jacobian、重力对照 |
| [check_piper_climb_observations.py](../mujoco/check_piper_climb_observations.py) | 完整 253 维原生/MuJoCo 观测对照 |

共享 runtime 增加可选的机器人命名、质心速度、末端几何和姿态 IK 参数；ARX5 默认值与原 actor 保留。PIPER 已有独立目标导航及技能检查，但尚未完成生产 executor、技能记录和移动箱高台组合集成。

## 5. 环境与产物

- 当前本地仓库：`/home/yuanyue/re-nav -piper-deploy/GO2-ARX5-MUJOCO`。不要混用原 `/home/yuanyue/re-nav` 或当前工作区的嵌套副本。
- 本地解释器：`/home/yuanyue/re-nav -piper-deploy/.envs/go2-arx5-nav/bin/python`；实测 MuJoCo 3.12.0、PyTorch 2.7.0+cpu、NumPy 2.4.6，AVX2，单线程。仅在此副本环境新增 `usd-core==25.11` 用于 CPU 资产解析。
- 本地产物：`/home/yuanyue/re-nav -piper-deploy/artifacts/go2-piper/2026-09-15/`，位于 Git 外。
- 最新索引：[piper-three-skills-index.json](../../artifacts/go2-piper/2026-09-15/piper-three-skills-index.json)，明确保存 `mujoco_all_skills_passed=false`、`hardware_validated=false`。
- PUSH 训练源码：`/mnt/yuanyue/data/go2-piper/2026-09-15/skills-v9-u8KMLk/`；选定 run 为 `logs/rsl_rl/go2_piper_box_push_hybrid/2026-09-15_18-24-25_piper_pose_v9/model_599.pt`。
- CLIMB 训练源码：`/mnt/yuanyue/data/go2-piper/2026-09-15/skills-v6-uk4pBB/`；选定 run 为 `logs/rsl_rl/go2_piper_box_climb/2026-09-15_17-50-11_piper_center_v6/model_200.pt`。
- 服务器评估及三份包归档：`/mnt/yuanyue/data/go2-piper/2026-09-15/eval-t8FO0a/`。其源码使用旧位置 IK，PUSH 姿态模型的原生重放应在 v9 源码下执行；不要仅凭包在此目录就误用该目录的 PUSH 代码。
- 原生运行继续使用既有 `/mnt/yuanyue/bin/isaac-python` 与隔离 `PYTHONPATH`。没有修改服务器原仓库或共享安装环境，没有推送分支。

三份本地包均包含策略和 manifest，模型不随 Git 自动取得：

| 技能 | 包 | policy SHA-256 |
| --- | --- | --- |
| NAV | [piper-nav-baseline-bundle](../../artifacts/go2-piper/2026-09-15/piper-nav-baseline-bundle/) | `72d9f22c633081e8d4b9693e6fda54dba6cdd036a102f330c1bfe096e43b4da8` |
| PUSH | [piper-push-pose599-bundle](../../artifacts/go2-piper/2026-09-15/piper-push-pose599-bundle/) | `c3ddb39280fbbfd15327153b5bfd001dbbc75a49d86505b1bded762f344fc83a` |
| CLIMB | [piper-climb-center200-bundle](../../artifacts/go2-piper/2026-09-15/piper-climb-center200-bundle/) | `6d25118145668e381de648eebbe57b5ffe7dd4eb781ede7d1bc341eb559ecc32` |

本地 `push-v9/`、`climb-v6/` 保存完整 checkpoint 和同 run 参数，可以继续训练。原 `native-nav-7CkC5q`、`skills-train-GOuORG` 等目录及早期 bootstrap 索引保留作历史，不能覆盖本文最终模型选择。

## 6. 最小复现

在当前本地仓库根目录执行，检查会拒绝覆盖已有报告：

```bash
python_bin="../.envs/go2-arx5-nav/bin/python"
artifacts="../artifacts/go2-piper/2026-09-15"
"$python_bin" mujoco/check_piper_runtime_contract.py
"$python_bin" mujoco/check_piper_robot_profile.py
check_dir=$(mktemp -d "$artifacts/recheck-XXXXXX")
"$python_bin" mujoco/check_piper_navigation.py --seeds 1 --seed-offset 500 --output-json "$check_dir/nav.json"
"$python_bin" mujoco/check_piper_box_skills.py --skill push --policy "$artifacts/piper-push-pose599-bundle/policy.pt" --push-ik-mode pose --native-collisions --seeds 1 --seed-offset 500 --output-json "$check_dir/push.json"
"$python_bin" mujoco/check_piper_box_skills.py --skill climb --policy "$artifacts/piper-climb-center200-bundle/policy.pt" --native-collisions --seeds 1 --seed-offset 500 --output-json "$check_dir/climb.json"
```

最后一项当前预期失败；批量箱体检查只要有失败便返回非零。不要删失败 seed、降低接触判据或把停止在箱前的动作计为登箱成功。

原生验收示例：先确认 GPU 空闲，在 PUSH v9 源码目录执行；CLIMB 在其 v6 源码下改用对应 task/checkpoint 和新输出路径。

```bash
result_dir=$(mktemp -d ./recheck-XXXXXX)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/source/LeggedManip_Lab" /mnt/yuanyue/bin/isaac-python scripts/evaluate_box_skills.py --headless --device cuda:0 --task GO2-PIPER-Box-Push-Hybrid-Play --checkpoint logs/rsl_rl/go2_piper_box_push_hybrid/2026-09-15_18-24-25_piper_pose_v9/model_599.pt --num_envs 32 --seed 206 --output-json "$result_dir/push.json"
```

原生评估脚本退出码表示脚本是否正常执行，技能是否达标必须读取 `successful_episodes`、`incomplete_episodes` 和终止原因。正式 PIPER 导出器还会验证两组不同 seed、每组至少 30/32、checkpoint/控制/资产 hash，以及没有诊断覆盖。

迁移到别处时改用实际路径，并重新验证导入来源与资产 hash；不根据历史 GPU 空闲记录启动作业。所有旧训练终端都应按已结束或已明确中止处理，不因旧 active 文本重复训练。

## 7. 下一步

1. 冻结三份选定包，优先解决 CLIMB 0/32 与 PUSH 21/32 的动态迁移差距。静态和观测对照已经通过，下一步对照逐步命令、力矩、延迟、接触求解和初始化；不要无依据地重新训练已通过原生验收的 actor。
2. 完成更宽起点、负载、摩擦和高度检查，再接生产 executor、快照/事件与移动箱高台组合。当前两个独立箱体技能通过原生门槛，不等于连续无 reset 组合已经完成。
3. 收集实物接口和观测清单，完成硬件限幅标定、只读反馈与离线推理，再按计划逐级上机；当前没有任何实机放行结论。
4. 本组后续提交和上传均使用 `feature/go2-piper-sim2real`。保留两个未跟踪 ARX5 中间权重，不上传环境、模型产物或凭据。

## 8. 训练历史边界

- 首轮 PIPER PUSH 400 更新和从 NAV 投影初始化的 CLIMB 1000 更新均未通过原生验收，报告保留；完成训练次数不等于可用技能。
- 后续 CLIMB 使用相同 253/18 契约的 ARX5 登箱 actor 初始化，在 PIPER 动力学下独立微调，最终按独立评估选择 v6 的 `model_200.pt`。ARX5 原接受权重没有被覆盖或重训。
- PUSH 去除静止接触正收益后逐渐出现推进，但旧位置 IK 版本 v5 的独立结果仅 5/32；最终采用可达倾角姿态 IK 下训练的 v9 `model_599.pt`。
- 水平姿态、缩短机身接近距离、持续位置 IK、CLIMB 固定机械臂、起步准备和观测噪声均做过诊断，不能将这些未通过配置与最终包混用。
- PUSH v4 的静止策略和 CLIMB v10 的退化精调已明确中止，其 checkpoint/log 保留；没有需要等待的旧训练任务。后续若续训需重新核对实际进程、模型和源代码。

## 9. 收尾检查

本轮八个聚焦检查脚本共 40 项用例通过，覆盖模型初始化、导出来源、PUSH 控制、奖励、评估配额、共享 runtime、PIPER runtime 和机器人 profile；改动 Python 文件编译通过，原 ARX5 NAV-CLIMB-NAV 切换回归 1/1 通过。三份交接/计划文档的本地链接及 `git diff --check` 通过。这些代码检查不覆盖上表中尚未通过的 MuJoCo 物理迁移门槛。

2026-09-15 收尾查询未发现本组训练或评估进程，GPU 查询也没有计算作业；这是本次查询快照，下一次使用服务器仍需重新检查。当前工作仍未提交或推送。