# GO2-ARX5 CLIMB 训练记录

最后验证日期：2026-09-10

## 当前结论

- 独立 Isaac 任务 `GO2-ARX5-Climb` 和 `GO2-ARX5-Climb-Play` 已实现。
- RTX 4090 上的 4096 环境正式训练已完成至迭代 1499。
- Isaac 原生评估表明最终策略能够在台阶课程中稳定运行并产生明显抬升。
- 最终 actor 已导出为 TorchScript 和 ONNX，并下载到本地。
- 最终 actor 已在本地 MuJoCo 低台阶场景中通过 20 秒完整穿越，CLIMB Sim-to-Sim 已通过当前阶段验收。
- CLIMB 已解除复杂任务 executor 的集成阻塞，但仍需在集成后补随机化回归。

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

## 远程位置

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
- Isaac `DelayedPDActuator` 的 physics-step 延迟和 reset 零目标缓冲。
- USD 腿部关节速度限制，包括小腿 `15.7 rad/s` 限制下的位置积分。
- CLIMB 训练配置中的机械臂 PD 增益和未裁剪 joint target。
- 使用 free-joint `qvel` 构造 body-frame root velocity observation。

根因是本地 runtime 未复现训练时的启动执行器语义，并且通过 `mj_objectVelocity` 得到了与 Isaac root velocity 不一致的第二帧观测。修复后，最终 actor 在固定低台阶场景中的 20 秒结果为：

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

## 下一步

1. 扩大 actuator delay、初始姿态、箱体位置和台阶高度随机化回归。
2. 从任意场景网格自动提取 PLATFORM entry/landing portal。
3. 增加 STOP/Recovery，处理技能失败后的站立恢复和局部重试。
4. 评估 JUMP 是否进入下一阶段；当前尚无可执行 Jump actor/runtime。
5. 保持当前跳跃/前扑式 CLIMB 动作风格，按任务成功和落稳验收。

当前不需要重新训练 CLIMB；后续工作转为复杂任务集成和泛化验证。