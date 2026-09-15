# yuanyue 双 RTX 4090 部署与开发记录

当前核对日期：2026-09-15；初始部署验收日期：2026-09-11。统一项目状态见 [主交接](AGENT_HANDOFF_CN.md)，本文负责机器、环境、路径和部署证据。

代码发布基线为 `008a7958cf542fe05a318290d4534358e0704f2b`，分支 `feature/reconfigurable-navigation-oracle`，2026-09-15已推送并核对GitHub。服务器Git HEAD仍为 `6c71d1a61dbf433022b4c06f0561344884225532`，工作树存在累计同步和资产改动；实际运行文件按源码/模型hash核对，不直接pull或reset。新权重、数据与录像不随Git发布。

后续进度：完整技能起点快照和跨进程重放已通过，新增 600 请求/542 条执行记录的候选 pilot；详见 [N1 快照与数据记录](N1_REPLAY_DATA_CN.md)。下文第 7 节保留最早 50 条边界记录的阶段性结果。

2026-09-12 更新：schema v2 控制步过程事件和技能失败原因已同步并在新服务器通过 32 项合约、复杂课程 10/10、CLIMB 切换 10/10、PUSH 5/5 和五个技能各三次跨进程重放。新增 60 请求为 44 成功、9 失败、7 拒绝；预算测试为 6 截断、3 拒绝，详见 [过程事件验收](N1_SKILL_EVENTS_CN.md)。数据保存在 `/mnt/yuanyue/data/n1-events-uDR6Is/`。SSH 公钥登录已通过全新无复用连接验证，本地别名为 `go2-yuanyue`。

同日 N2 更新：参数化通道和扫描已同步，累计 40 项合约及原默认回归通过；十类单变量场景得到 7 成功、2 执行失败、1 规则拒绝，生成 28 起点和 84 候选，详见 [参数化通道验收](N2_PASSAGE_SCENES_CN.md)。新数据为 `/mnt/yuanyue/data/n2-passages-aoEpYv/`，仍只有一个布局族，不是泛化评测。

## 1. 当前结论

- 新服务器的 mihomo、隔离 MuJoCo 环境和隔离 Isaac 环境已经部署。
- MuJoCo旧基线及逐卡PPO冒烟已于初始部署阶段完成；后续两卡并行独立训练也已完成，不能再标记为“并行未验证”。这不等于DDP或双卡加速比验收。
- 最新自由5 kg、摩擦0.4、20 cm箱到40 cm高台组合，在seed500-531上本地和服务器均29/32，失败510/525/528均为高台机身接触；需要显式AVX2启动入口，详见 [箱体训练记录](BOX_SKILL_TRAINING_CN.md)。
- 原接受的actor保留，新三份候选独立保存；高台actor原生21/32、22/32仍未过单策略门槛，组合分数包含NAV对位和顶面定位，尚未接入生产executor。
- N1 已完成可选边界记录、技能起点物理/控制器快照、归档重放、控制步事件及失败原因和参数扰动物理候选 pilot。逐物理步覆盖、完整失稳/支撑判据、多场景划分和正式训练数据集仍待完成。
- 三轮连续起步/场景/间隙候选训练均已结束，原生录像评估与完整MuJoCo录像已完成。2026-09-15检查未发现训练或评估进程，两张GPU各0%利用率、1 MiB占用；新作业前重查。
- 当前下一步是收敛剩余高台接触失败与正式集成，世界模型训练和大规模数据扩容暂缓。本文更新没有重新运行策略批量验收。

## 2. 环境与路径

| 项目 | 实际状态 |
| --- | --- |
| 系统 | Ubuntu 22.04.4 LTS，x86_64 |
| GPU / 驱动 | 2 x RTX 4090，每卡 24564 MiB，580.82.07 |
| 主机资源 | 约 114 GiB RAM；容器可用 CPU 配额 32 |
| 持久化目录 | `/mnt/yuanyue`，JuiceFS；显示的共享池容量不是个人配额，也不是已验证的本地 NVMe 性能 |
| 项目源码 | `/mnt/yuanyue/GO2-ARX5-MUJOCO` |
| 当前SSH入口 | `go2-yuanyue`，端口46308；公钥登录已配置 |
| 服务器Git状态 | HEAD `6c71d1a`，工作树不干净；发布代码基线为008a795 |
| Micromamba | `/mnt/yuanyue/bin/micromamba`，2.9.0 |
| MuJoCo 环境 | `/mnt/yuanyue/envs/go2-mujoco` |
| Isaac 环境 | `/mnt/yuanyue/envs/go2-isaac` |
| Isaac Lab | `/mnt/yuanyue/IsaacLab`，`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` |
| Isaac 启动器 | `/mnt/yuanyue/bin/isaac-python` |
| 最新模型包、报告与录像 | `/mnt/yuanyue/data/box-skills-eval/` |
| 训练文本日志 | `/mnt/yuanyue/logs/skill-retraining/` |

2026-09-15包元数据核对：MuJoCo环境为Python3.11.16、MuJoCo3.12.0、PyTorch2.7.0+cpu、NumPy2.4.6、imageio2.37.4、imageio-ffmpeg0.6.0，`pip check` 通过。Isaac环境为Python3.11.16、Isaac Sim5.1.0.0、Isaac Lab0.54.4、RSL-RL5.0.1、PyTorch2.7.0+cu128、NumPy1.26.0；GPU仍为两张RTX4090，驱动580.82.07。

新 Isaac 环境只剩一个已知元数据冲突：Isaac Sim 固定的 `fastapi==0.115.7` 要求 `starlette<0.46.0`，而固定的 Isaac Lab 要求 `starlette==0.49.1`。本轮没有为此修改上游源码或 GPU 驱动；真实训练已通过，但不能宣称依赖完全无冲突或相关 Web 服务已验证。

2026-09-15重新执行Isaac `pip check` 仍仅报告上述冲突；本次只更新文档，未调整依赖。

## 3. 下载代理

mihomo v1.19.30 由 runit 管理，实际服务目录为 `/opt/devmachine/init/service/go2-mihomo`。容器 PID 1 监控 `/opt/devmachine/init/service`，不是默认 `/etc/service`。

- HTTP：`127.0.0.1:7890`；SOCKS：`127.0.0.1:7891`；mixed：`127.0.0.1:7893`。
- 控制接口只监听 `127.0.0.1:9090` 并启用密钥；TUN 关闭。
- 订阅与节点配置只放在服务器私有目录，文件权限为 `600`，本文不记录订阅 URL、节点密码或控制密钥。
- 安装命令按需设置代理变量，没有改动全局路由或 SSH 授权密钥。
- 2026-09-15发布008a795时，直连GitHub超时，使用临时 `127.0.0.1:17890 -> 服务器127.0.0.1:7890` SSH转发完成推送，并核对远端hash。该转发已关闭，不要假定仍存在或重复修改全局Git代理。
- 实测 PyPI 680355 字节索引约 1.3 秒下载完成；176 MB PyTorch CPU wheel 下载约 13 MB/s。这不是固定带宽承诺。

服务器上检查服务：

```bash
sv status /opt/devmachine/init/service/go2-mihomo
```

若容器被重建，持久化程序和配置可以保留，但需要根据新的 PID 1 监控目录重新注册服务。

## 4. 初始部署验收（2026-09-11历史记录）

以下结果对应原有低摩擦PUSH和低台阶CLIMB基线，不是新箱体流程的当前成功率，也不是2026-09-15重新执行的结果。

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

该双卡容器中，保持全部GPU可见，同时显式指定仿真、策略和渲染设备。下面保留重建或排错所用的最小冒烟示例；现有环境已验证，不因阅读交接而自动重跑或开始训练：

```bash
cd /mnt/yuanyue/GO2-ARX5-MUJOCO
env -u CUDA_VISIBLE_DEVICES /mnt/yuanyue/bin/isaac-python scripts/rsl_rl/train.py --task GO2-ARX5-Climb --device cuda:0 --num_envs 16 --seed 0 --max_iterations 1 --headless --logger tensorboard --experiment_name go2_migration_smoke --run_name gpu0_checked --kit_args="--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false" agent.device=cuda:0
env -u CUDA_VISIBLE_DEVICES /mnt/yuanyue/bin/isaac-python scripts/rsl_rl/train.py --task GO2-ARX5-Climb --device cuda:1 --num_envs 16 --seed 0 --max_iterations 1 --headless --logger tensorboard --experiment_name go2_migration_smoke --run_name gpu1_checked --kit_args="--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false" agent.device=cuda:1
```

只指定 `--device cuda:1` 会设置环境设备，但当前训练脚本的非分布式路径不会因此自动修改 `agent_cfg.device`；因此需要同步传入 `agent.device=cuda:1`。不要把适用于纯 PyTorch 的单卡掩码经验直接当成 Omniverse/Vulkan 的设备映射规则。

## 6. 模型与训练产物

最新组合所需包：`push-height020-stop200-bundle`、`climb-prepared-ground499-bundle`、`climb-prepared-gaps399-bundle`，均位于 `/mnt/yuanyue/data/box-skills-eval/`，不在Git。完整索引为 `continuous-climb-validation-v2.json`；最终录像 `box-to-platform-final-seed500.mp4`（约29.70s、743帧、25 FPS），同目录保留JSON和预览。新模型需要包内参数、起步分布和manifest，不应只带走一个 `.pt`。

复现入口为 [run_box_support_sequence.py](../mujoco/run_box_support_sequence.py)，只对子进程统一AVX2、MKL、oneDNN与单线程设置。默认自动指令路径曾使同一权重在本地得到29/32、服务器27/32；显式入口后两边均29/32且逐回合结局一致。命令和边界见 [主交接](AGENT_HANDOFF_CN.md)。

以下是迁移保留的旧低台阶CLIMB checkpoint：

已从旧服务器复制 `model_1499.pt`、`params/env.yaml` 和 `params/agent.yaml` 到：

```text
/mnt/yuanyue/GO2-ARX5-MUJOCO/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/
```

checkpoint 两端 SHA-256 一致：`9206fd87d72377c42d78334938d050be08067ad488eeda7fcaa25ebdf5c68674`。此次没有同步所有历史 TensorBoard events、视频和其他中间 checkpoint；旧服务器原文件保持不变。

## 7. N1 首步交付（历史记录）

- 新增 [SkillTransition](../mujoco/reconfigurable_navigation/data/transition.py) 和 executor 的可选 `on_transition` 回调，旧 `SkillExecutionRecord` 和默认运行接口保持兼容。
- 记录动作、前后观测的独立拷贝、前序技能、状态、仿真开始/结束时间及中断标志。CLIMB 的 0.5 秒准备阶段计入总耗时。
- JSON 使用 schema_version=1，支持 NumPy 和枚举，拒绝 NaN/Infinity；中断样本的 skill_success 为 null，不伪造物理失败标签。
- [合约测试](../mujoco/check_skill_transitions.py) 在本地和新服务器均 7/7 通过。新机带记录的复杂课程 10/10，默认未记录的 PUSH/CLIMB 切换回归也通过。
- 实际输出 50 条记录，48 条 succeeded、2 条 failed。任务最终成功不代表每次技能成功，数据保留原始技能状态。

数据样例位于服务器：

```text
/mnt/yuanyue/data/n1-boundary/complex-course-twpEPA.jsonl
```

这份50条数据本身只是10个固定课程episode的边界验证，不是完整pilot或正式训练集；后续另有600请求/542执行的快照候选pilot及N2通道数据，见文首链接。它们仍不能替代多场景族泛化和正式数据划分。

在新服务器生成新文件：

```bash
cd /mnt/yuanyue/GO2-ARX5-MUJOCO
/mnt/yuanyue/bin/micromamba run -p /mnt/yuanyue/envs/go2-mujoco python mujoco/check_complex_course.py --seeds 10 --record-jsonl logs/n1-transitions.jsonl
```

该边界 JSONL 选项采用单进程追加写入，不同并行 worker 应使用不同文件。后续快照/候选和事件/原因功能见开头链接；完整支撑时序判据、多场景族和正式 train/validation/test 划分仍待实现。