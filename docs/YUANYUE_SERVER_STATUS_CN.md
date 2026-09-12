# yuanyue 双 RTX 4090 部署与开发记录

验证日期：2026-09-11。部署源码基线：`6c71d1a`，分支：`feature/reconfigurable-navigation-oracle`。后续 N1 代码已同步到新服务器，发布版本以 Git 历史为准；数据、快照和训练产物保留在服务器。

后续进度：完整技能起点快照和跨进程重放已通过，新增 600 请求/542 条执行记录的候选 pilot；详见 [N1 快照与数据记录](N1_REPLAY_DATA_CN.md)。下文第 7 节保留最早 50 条边界记录的阶段性结果。

## 1. 当前结论

- 新服务器的 mihomo、隔离 MuJoCo 环境和隔离 Isaac 环境已经部署。
- MuJoCo 原有基线全部通过；两张 GPU 分别完成 `16 env x 1 PPO iteration` 的真实 Isaac 验收并生成 checkpoint。
- 原 CLIMB 的可续训 checkpoint 和训练参数已迁移，最终部署 actor 未修改或重新训练。
- N1 已完成可选边界记录、技能起点物理/控制器快照、归档重放及参数扰动物理候选 pilot。技能内部事件标签、多场景划分和正式训练数据集仍待完成。
- 双卡同时训练、DDP、完整 checkpoint 的原生录像评估尚未在新机验收。

## 2. 环境与路径

| 项目 | 实际状态 |
| --- | --- |
| 系统 | Ubuntu 22.04.4 LTS，x86_64 |
| GPU / 驱动 | 2 x RTX 4090，每卡 24564 MiB，580.82.07 |
| 主机资源 | 约 114 GiB RAM；容器可用 CPU 配额 32 |
| 持久化目录 | `/mnt/yuanyue`，JuiceFS；显示的共享池容量不是个人配额，也不是已验证的本地 NVMe 性能 |
| 项目源码 | `/mnt/yuanyue/GO2-ARX5-MUJOCO` |
| Micromamba | `/mnt/yuanyue/bin/micromamba`，2.9.0 |
| MuJoCo 环境 | `/mnt/yuanyue/envs/go2-mujoco` |
| Isaac 环境 | `/mnt/yuanyue/envs/go2-isaac` |
| Isaac Lab | `/mnt/yuanyue/IsaacLab`，`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` |
| Isaac 启动器 | `/mnt/yuanyue/bin/isaac-python` |

MuJoCo 环境为 Python 3.11.16、MuJoCo 3.12.0、PyTorch 2.7.0+cpu、NumPy 2.4.6，`pip check` 通过。Isaac 环境为 Python 3.11.16、Isaac Sim 5.1.0.0、Isaac Lab 0.54.4、isaaclab_rl 0.5.2、RSL-RL 5.0.1、PyTorch 2.7.0+cu128、NumPy 1.26.0。

新 Isaac 环境只剩一个已知元数据冲突：Isaac Sim 固定的 `fastapi==0.115.7` 要求 `starlette<0.46.0`，而固定的 Isaac Lab 要求 `starlette==0.49.1`。本轮没有为此修改上游源码或 GPU 驱动；真实训练已通过，但不能宣称依赖完全无冲突或相关 Web 服务已验证。

## 3. 下载代理

mihomo v1.19.30 由 runit 管理，实际服务目录为 `/opt/devmachine/init/service/go2-mihomo`。容器 PID 1 监控 `/opt/devmachine/init/service`，不是默认 `/etc/service`。

- HTTP：`127.0.0.1:7890`；SOCKS：`127.0.0.1:7891`；mixed：`127.0.0.1:7893`。
- 控制接口只监听 `127.0.0.1:9090` 并启用密钥；TUN 关闭。
- 订阅与节点配置只放在服务器私有目录，文件权限为 `600`，本文不记录订阅 URL、节点密码或控制密钥。
- 安装命令按需设置代理变量，没有改动全局路由或 SSH 授权密钥。
- 实测 PyPI 680355 字节索引约 1.3 秒下载完成；176 MB PyTorch CPU wheel 下载约 13 MB/s。这不是固定带宽承诺。

服务器上检查服务：

```bash
sv status /opt/devmachine/init/service/go2-mihomo
```

若容器被重建，持久化程序和配置可以保留，但需要根据新的 PID 1 监控目录重新注册服务。

## 4. 新机验收结果

| 检查 | 实际结果 |
| --- | --- |
| Oracle 结构检查 | 100/100 |
| NAV 物理回归 | 100/100 |
| NAV-PUSH-NAV | 20/20 |
| NAV-CLIMB-NAV | 10/10 |
| 完整复杂课程 | 10/10 |
| 独立 CLIMB | 20 秒，x 位移 8.16 m，最低 base 高度 0.21 m，fell=False |
| GPU 0 Isaac/PPO | 16 环境，384 个环境步，1 次迭代，checkpoint 写入成功 |
| GPU 1 Isaac/PPO | 同样完成；日志确认仿真与渲染选择 GPU 1，策略配置也显式设为 cuda:1 |

MuJoCo 日志在 `/mnt/yuanyue/logs/regressions/`。两次通过的 Isaac 日志在 `/mnt/yuanyue/logs/isaac-smoke/gpu0_checked.log` 和 `gpu1_checked.log`，均未发现 `[Error]`、Traceback、CUDA error 或 Fatal 标记。它们是单卡串行冒烟，不是双卡加速或训练质量实验。

两个冒烟 checkpoint：

```text
logs/rsl_rl/go2_migration_smoke/2026-09-11_17-47-27_gpu0_checked/model_0.pt
logs/rsl_rl/go2_migration_smoke/2026-09-11_17-50-59_gpu1_checked/model_0.pt
```

初次 GPU 0 启动发现缺少 `libGLU.so.1`，同时单卡 `CUDA_VISIBLE_DEVICES` 掩码触发 Omniverse 枚举警告，未通过验收；该测试进程在常规终止无效后被定向终止，退出码 137 是清理行为，不是安装失败。补齐 `libglu1-mesa` 并改用显式设备编号后，两卡分别通过。旧失败日志保留，没有覆盖或算入通过结果。

## 5. 已验证的 GPU 启动方式

该双卡容器中，保持全部 GPU 可见，同时显式指定仿真、策略和渲染设备。下面命令均在新服务器执行，仍是独立的最小冒烟，不会替换最终 actor：

```bash
cd /mnt/yuanyue/GO2-ARX5-MUJOCO
env -u CUDA_VISIBLE_DEVICES /mnt/yuanyue/bin/isaac-python scripts/rsl_rl/train.py --task GO2-ARX5-Climb --device cuda:0 --num_envs 16 --seed 0 --max_iterations 1 --headless --logger tensorboard --experiment_name go2_migration_smoke --run_name gpu0_checked --kit_args="--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false" agent.device=cuda:0
env -u CUDA_VISIBLE_DEVICES /mnt/yuanyue/bin/isaac-python scripts/rsl_rl/train.py --task GO2-ARX5-Climb --device cuda:1 --num_envs 16 --seed 0 --max_iterations 1 --headless --logger tensorboard --experiment_name go2_migration_smoke --run_name gpu1_checked --kit_args="--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false" agent.device=cuda:1
```

只指定 `--device cuda:1` 会设置环境设备，但当前训练脚本的非分布式路径不会因此自动修改 `agent_cfg.device`；因此需要同步传入 `agent.device=cuda:1`。不要把适用于纯 PyTorch 的单卡掩码经验直接当成 Omniverse/Vulkan 的设备映射规则。

## 6. 原始训练产物

已从旧服务器复制 `model_1499.pt`、`params/env.yaml` 和 `params/agent.yaml` 到：

```text
/mnt/yuanyue/GO2-ARX5-MUJOCO/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/
```

checkpoint 两端 SHA-256 一致：`9206fd87d72377c42d78334938d050be08067ad488eeda7fcaa25ebdf5c68674`。此次没有同步所有历史 TensorBoard events、视频和其他中间 checkpoint；旧服务器原文件保持不变。

## 7. N1 首步交付

- 新增 [SkillTransition](../mujoco/reconfigurable_navigation/data/transition.py) 和 executor 的可选 `on_transition` 回调，旧 `SkillExecutionRecord` 和默认运行接口保持兼容。
- 记录动作、前后观测的独立拷贝、前序技能、状态、仿真开始/结束时间及中断标志。CLIMB 的 0.5 秒准备阶段计入总耗时。
- JSON 使用 schema_version=1，支持 NumPy 和枚举，拒绝 NaN/Infinity；中断样本的 skill_success 为 null，不伪造物理失败标签。
- [合约测试](../mujoco/check_skill_transitions.py) 在本地和新服务器均 7/7 通过。新机带记录的复杂课程 10/10，默认未记录的 PUSH/CLIMB 切换回归也通过。
- 实际输出 50 条记录，48 条 succeeded、2 条 failed。任务最终成功不代表每次技能成功，数据保留原始技能状态。

数据样例位于服务器：

```text
/mnt/yuanyue/data/n1-boundary/complex-course-twpEPA.jsonl
```

这是 10 个固定课程分布 episode 的合约验证数据，不是 500-1000 条 pilot 已完成，也不是可泛化的正式训练集。

在新服务器生成新文件：

```bash
cd /mnt/yuanyue/GO2-ARX5-MUJOCO
/mnt/yuanyue/bin/micromamba run -p /mnt/yuanyue/envs/go2-mujoco python mujoco/check_complex_course.py --seeds 10 --record-jsonl logs/n1-transitions.jsonl
```

该边界 JSONL 选项采用单进程追加写入，不同并行 worker 应使用不同文件。后续快照/候选功能见开头链接；原因分类、碰撞/支撑时序标签、多场景族和正式 train/validation/test 划分仍待实现。