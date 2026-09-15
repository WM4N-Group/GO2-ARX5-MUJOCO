# GO2-ARX5 低台阶 CLIMB 历史训练记录

文档范围更新：2026-09-15；下述实验验证日期：2026-09-10。本次只整理交接边界，没有重跑下述实验。

本文记录旧 `GO2-ARX5-Climb` 的低台阶策略 `policy_iter1499.pt`，不是当前移动箱高台候选。当前代码发布基线008a795、机器状态和后续任务见 [主交接](AGENT_HANDOFF_CN.md)，新的单箱、连续起步及间隙训练见 [箱体训练记录](BOX_SKILL_TRAINING_CN.md)。当前完整自由箱到40 cm高台流程29/32，使用三份非Git模型包，尚未接入生产executor；高台actor单独原生验收仍未达标。

当前服务器为 `go2-yuanyue`（端口46308、`/mnt/yuanyue`）；下述 `go2-4090` 和 `/mnt/miaojigui` 仅用于追溯历史。旧checkpoint及参数已迁移到 `/mnt/yuanyue/GO2-ARX5-MUJOCO/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/`，不必重新训练该接受基线。

## 低台阶基线结论

- 独立 Isaac 任务 `GO2-ARX5-Climb` 和 `GO2-ARX5-Climb-Play` 已实现。
- RTX 4090 上的 4096 环境正式训练已完成至迭代 1499。
- Isaac 原生评估表明最终策略能够在台阶课程中稳定运行并产生明显抬升。
- 最终 actor 已导出为 TorchScript 和 ONNX，并下载到本地。
- 最终 actor 已在本地 MuJoCo 低台阶场景中通过 20 秒完整穿越，CLIMB Sim-to-Sim 已通过当前阶段验收。
- 该旧CLIMB已接入生产executor，后续切换和旧复杂课程各10/10；这不代表新的移动箱支撑流程已生产集成。

## 任务设计

配置文件：

- `source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/climb_env_cfg.py`
- `source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/agents/rsl_rl_ppo_cfg.py`
- `source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/go2_arx5/__init__.py`

训练定义：

- 地形：倒金字塔台阶课程。
- 台阶高度：`0.02-0.12 m`。
- 台阶宽度：`0.35 m`。
- Terrain rows/columns：`8 x 8`。
- 初始最大 terrain level：`1`。
- 动作：18 维关节位置 residual。
- Actor 观测：253 维。
- Height scan：187 点，`1.6 x 1.0 m`，分辨率 `0.1 m`。
- 指令：仅前进速度，`vx=0.4-0.6 m/s`。
- Actor/critic observation group：显式使用 `policy`。
- 训练日志名：`go2_arx5_climb`。

## 原训练位置（旧单卡服务器）

- 训练主机：OpenSSH alias `go2-4090`。
- 训练 worktree：`/mnt/miaojigui/worktrees/go2-climb`。
- Isaac 启动器：`/mnt/miaojigui/bin/isaac-python`。
- 最终 run：`/mnt/miaojigui/worktrees/go2-climb/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49`。
- 最终 checkpoint：上述目录中的 `model_1499.pt`。
- 正式训练日志：`/mnt/miaojigui/worktrees/go2-climb/climb_stage2_to_1500.log`。
- 最终评估日志：`/mnt/miaojigui/worktrees/go2-climb/climb_final_eval.log`。

## 训练过程

训练分为以下阶段：

1. `16 env x 1 iteration`：验证任务注册、terrain、RayCaster、manager 和 checkpoint。
2. `256 env x 50 iterations`：从零训练诊断，生成 `model_49.pt`。
3. `64 env` 从迭代 49 续训到约 348：确认 reward 和存活时间改善。
4. `4096 env x 1 iteration`：验证 RTX 4090 显存容量。
5. `4096 env` 从 `model_348.pt` 续训到 `model_1499.pt`。

正式阶段耗时约 46 分钟。最终指标：

```text
iteration: 1499/1500
mean_reward: 65.25
mean_episode_length: 979.29 / 1000
terrain_level: 4.6864 / 7
base_contact_termination_metric: 0.0201
velocity_xy_error: 0.1324
action_std: 0.38
```

关键趋势：

| Iteration | Reward | Episode length | Terrain level | Base contact |
| ---: | ---: | ---: | ---: | ---: |
| 350 | 1.06 | 66.04 | 1.3411 | 0.1044 |
| 500 | 51.63 | 975.48 | 2.5791 | 0.0261 |
| 750 | 57.68 | 939.62 | 4.1286 | 0.0467 |
| 1000 | 62.59 | 976.16 | 4.8747 | 0.0466 |
| 1250 | 65.18 | 989.65 | 4.7482 | 0.0327 |
| 1499 | 65.25 | 979.29 | 4.6864 | 0.0201 |

## Isaac 原生评估

最终 JIT 在 `GO2-ARX5-Climb-Play` 中运行 16 个环境、1000 步：

```text
16/16 环境最大行进距离超过 3 m
平均最大行进距离：9.76 m
平均最大抬升：0.107 m
最大抬升：0.414 m
5/16 环境抬升超过 0.10 m
每个环境只有 1 次 episode 结束，对应完整 1000 步 horizon
```

Play 产物：

- 本地跟随视频：`/home/yuanyue/re-nav/artifacts/go2-climb/iter1499/climb-follow-camera.mp4`
- 本地 ONNX：`/home/yuanyue/re-nav/artifacts/go2-climb/iter1499/policy.onnx`
- 本地 JIT：`mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt`
- 中间 JIT：`policy_iter348.pt`、`policy_iter900.pt`，仅用于诊断。

## 本地 MuJoCo 诊断

场景和入口：

- `mujoco/train/go2_arx5/climb_scene.xml`
- `mujoco/visualize_climb_policy.py`

运行命令：

```bash
cd /home/yuanyue/re-nav/GO2-ARX5-MUJOCO
/home/yuanyue/re-nav/.tools/micromamba run \
  -p /home/yuanyue/re-nav/.envs/go2-arx5-nav \
  python mujoco/visualize_climb_policy.py \
  --policy mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt \
  --duration 20
```

本地 runtime 已对齐：

- 253 维 observation 顺序。
- 187 点 yaw-aligned height scan。
- 初始 base 高度 `0.55 m`。
- `joint2/joint3` soft-limit 初态 `0.15 rad`。
- 策略从 reset 立即接管，不使用 Flat demo 的 3 秒 hold。
- Isaac policy/action 关节顺序和 MuJoCo 关节映射。
- 旧部署基线保留的physics-step延迟和reset零目标缓冲契约。新Box任务已按当前Isaac实现改为首命令填满缓冲，不能把本文旧契约套到新Box runtime。
- USD 腿部关节速度限制，包括小腿 `15.7 rad/s` 限制下的位置积分。
- CLIMB 训练配置中的机械臂 PD 增益和未裁剪 joint target。
- 使用 free-joint `qvel` 构造 body-frame root velocity observation。

当时诊断集中于旧runtime的启动契约及通过 `mj_objectVelocity` 构造的第二帧root velocity不一致。下表是该旧profile的实测结果，不构成当前Isaac缓冲实现的通用定义；新Box profile还分别修正了训练资产惯量、安装位置、法向力和首目标缓冲，见箱体训练记录。

| Command | X displacement | Final base height | Min base height | Base contact |
| ---: | ---: | ---: | ---: | ---: |
| `0.4 m/s` | `6.56 m` | `0.37 m` | `0.21 m` | no |
| `0.5 m/s` | `8.16 m` | `0.23 m` | `0.21 m` | no |
| `0.6 m/s` | `9.82 m` | `0.28 m` | `0.21 m` | no |

默认 `0.5 m/s` 运行中，机器人在台阶区间的 base 高度由约 `0.23 m` 上升到 `0.41-0.42 m`，并越过完整台阶和顶层平台。

## 技能切换集成

- `ClimbRuntime` 已抽离到 `mujoco/reconfigurable_navigation/climb_runtime.py`。
- `ClimbSkill` 已实现进入朝向、目标位置/高度、连续稳定和超时判定。
- `ReconfigurableExecutor` 可在共享 `MjModel/MjData` 上切换 NAV 与 CLIMB actor。
- 切换只初始化策略内部状态，不调用 `mj_resetData()`。
- `mujoco/check_climb_skill_switching.py --seeds 10` 已通过 `10/10`。
- 每个 episode 的物理 reset 计数均为 1，压缩后的执行阶段均为 `NAV -> CLIMB -> NAV`。
- executor 在进入 CLIMB 前显式保持默认姿态 `0.5 s`，消除 NAV/PUSH 遗留的机械臂姿态。
- 生产 `OraclePlanner` 已根据 PLATFORM entry/landing portal 自动生成 CLIMB。
- `mujoco/check_complex_course.py --seeds 10` 的真实 `NAV -> PUSH -> NAV -> CLIMB -> NAV` 课程已通过 `10/10`。

## 原后续方向（历史）

1. 扩大 actuator delay、初始姿态、箱体位置和台阶高度随机化回归。
2. 从任意场景网格自动提取 PLATFORM entry/landing portal。
3. 增加 STOP/Recovery，处理技能失败后的站立恢复和局部重试。
4. 评估 JUMP 是否进入下一阶段；当前尚无可执行 Jump actor/runtime。
5. 保持当前跳跃/前扑式 CLIMB 动作风格，按任务成功和落稳验收。

上述方向是低台阶阶段的安排，不是当前任务队列。当前优先收敛seed510/525/528的高台接触失败、验证更宽工况并准备新流程集成；JUMP与世界模型工作继续暂缓。原接受的低台阶/20 cm策略保持基线，新增能力通过独立候选验证。