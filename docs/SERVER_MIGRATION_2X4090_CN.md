# GO2-ARX5-MUJOCO 双 RTX 4090 服务器迁移指南

更新日期：2026-09-11。代码基线：`629e666`，分支：`feature/reconfigurable-navigation-oracle`。

本文用于把当前可重构导航项目迁移到一台配有两块 RTX 4090 的 Linux 服务器。目标是先复现 MuJoCo 推理与技能组合，再恢复 Isaac Lab 训练环境；迁移本身不要求重新训练策略。

## 1. 依赖边界

项目包含三条不同的运行路径，不能仅靠根目录的 `pip install -r requirements.txt` 安装全部环境。

| 运行路径 | 必需组件 | GPU 要求 |
| --- | --- | --- |
| MuJoCo NAV/PUSH/CLIMB 推理、Oracle 规划、复杂课程回归 | Python、MuJoCo、PyTorch、NumPy、SciPy、PyYAML 等 | 无头物理回归可使用 CPU；交互 Viewer 需要图形环境 |
| Isaac Lab 强化学习训练与原生评估 | NVIDIA 驱动、CUDA 版 PyTorch、Isaac Sim、Isaac Lab、RSL-RL、项目扩展及 USD 资产 | 单块 RTX 4090 已有成功记录；新服务器双卡配置需单独验收 |
| 可选 MuJoCo PPO 训练 | MuJoCo 环境加 Gymnasium、Stable-Baselines3、TensorBoard | 与 Isaac Lab/RSL-RL 是不同训练入口，不是运行现有技能的前置条件 |

建议建立两个隔离环境：`go2-mujoco` 用于 CPU 推理与回归，`go2-isaac` 用于 GPU 训练。不要为了 MuJoCo 回归在 Isaac 环境里安装 CPU 版 PyTorch，从而覆盖 CUDA 版本。

### 1.1 仓库安装清单的实际覆盖范围

- [根目录 requirements.txt](../requirements.txt)：`mujoco>=3.0.0,<4.0.0`、`pynput>=1.7`、`scipy>=1.10`、`pyyaml>=6.0`、`psutil>=5.9`、`prettytable>=3.0`；没有声明 PyTorch、Isaac Sim、Isaac Lab 或 RSL-RL。
- [项目扩展 setup.py](../source/LeggedManip_Lab/setup.py)：Python 下限为 `3.10`，运行依赖只声明 `psutil`。Editable 安装本项目不会自动安装完整训练栈。
- [可选 MuJoCo PPO requirements.txt](../mujoco/train/go2_arx5/requirements.txt)：`gymnasium>=1.0,<2.0`、`stable-baselines3>=2.6,<3.0`、`tensorboard>=2.15`。

因此，安装清单中的宽版本范围不等于已经验收过的版本组合。首次迁移优先复现下表中的已验证版本，再单独评估升级。

## 2. 已验证环境基线

以下是原开发机和原单卡训练服务器的历史验证记录，不表示已经在新双卡服务器执行过验证。

| 组件 | MuJoCo 开发/推理环境 | 原 RTX 4090 训练环境 |
| --- | --- | --- |
| 操作系统 | Linux，本地 Ubuntu 工作站 | Ubuntu 22.04.4 LTS，x86_64 |
| Python | 3.11.16 | 3.11.16 |
| MuJoCo | 3.12.0 | 3.12.0，作为独立验证工具 |
| PyTorch | 2.7.0+cpu | 2.7.0+cu128 |
| Isaac Sim | 不需要 | 5.1.0.0 |
| Isaac Lab | 不需要 | 源码 editable 安装，包版本 0.54.4，commit 见下文 |
| RSL-RL | 不需要 | rsl-rl-lib 5.0.1 |
| NVIDIA 驱动 | 无头 CPU 回归不需要 | 580.82.07 |
| GPU | CPU 即可运行当前回归 | RTX 4090，24 GB，计算能力 8.9 |
| 环境管理 | Micromamba | Micromamba |

2026-09-11 已只读查询旧训练服务器，确认 Isaac Lab 工作树干净，精确 commit 为：

```text
b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
```

注意：`Isaac Lab 0.54.4` 只是包版本，不能唯一确定 Git 提交。首次迁移固定以上 commit，不直接拉取最新 `main`，也不要换用 `v2.3.2`。当前项目使用 `RslRlMLPModelCfg` 等较新的配置 API；虽然训练入口的版本检查下限是 `3.0.1`，本项目配置的已验证组合是 `rsl-rl-lib==5.0.1`。驱动 `580.82.07` 是已成功运行的版本，不是本文声明的最低版本。

当前可执行技能为 NAV、PUSH、CLIMB，STOP 为终态。JUMP 尚无完整 actor/runtime/skill，不属于可迁移的已完成技能。

### 2.1 关键 Python 包的实测版本

下表来自本次本地及旧服务器的包元数据查询。它不是覆盖所有传递依赖的 lockfile。

| 包 | MuJoCo 环境 | Isaac 环境 |
| --- | --- | --- |
| numpy | 2.4.6 | 1.26.0 |
| scipy | 1.17.1 | 1.15.3 |
| PyYAML | 6.0.3 | 6.0.2 |
| pynput | 1.8.2 | 1.8.2 |
| psutil | 7.2.2 | 7.2.2，存在元数据冲突，见下文 |
| prettytable | 3.18.0 | 3.3.0 |
| LeggedManip_Lab | 0.1.0，editable | 0.1.0，editable |
| torchvision | 当前回归不需要 | 0.22.0+cu128 |
| gymnasium | 当前回归不需要 | 1.2.1 |
| isaaclab_assets / isaaclab_rl / isaaclab_tasks | 不需要 | 0.2.4 / 0.5.2 / 0.11.16 |
| tensordict | 不需要 | 0.14.1 |
| hydra-core / omegaconf | 不需要 | 1.3.6 / 2.3.1 |
| warp-lang / trimesh | 不需要 | 1.17.0 / 4.5.1 |
| tensorboard | 可选 | 2.21.0 |
| onnx / onnxscript | TorchScript 推理不需要 | 1.22.0 / 0.7.1，用于导出 |
| imageio / imageio-ffmpeg / moviepy | 可选 | 2.37.0 / 0.6.0 / 2.2.1，用于录像 |
| opencv-python-headless | 可选 | 4.11.0.86，用于视频检查 |

Isaac Lab 在此 commit 下要求 `numpy<2`，不能直接复用 MuJoCo 环境的 NumPy 2.x。ROS/ROS 2、实机 SDK、LiveAgent、VS Code、VPN 客户端都不是当前无头仿真训练的必需依赖；Isaac Sim 的完整安装可能带有 ROS 扩展，但不要求另行配置 ROS 工作空间。

### 2.2 旧 Isaac 环境已知依赖告警

本次查询旧服务器的 `pip check` 返回非零，报告以下四项：

| 已安装组件 | 声明要求 | 旧环境实际值 |
| --- | --- | --- |
| wheel 0.48.0 | packaging >=24.0 | packaging 23.0 |
| fastapi 0.115.7 | starlette >=0.40.0,<0.46.0 | starlette 0.49.1 |
| isaacsim-kernel 5.1.0.0 | psutil ==5.9.8 | psutil 7.2.2 |
| isaacsim-kernel 5.1.0.0 | typing_extensions ==4.12.2 | typing-extensions 4.16.0 |

这些告警不改变原有 NAV/PUSH/CLIMB 实测通过的事实，但说明旧环境不是零冲突的依赖解。部分约束来自不同组件，例如锁定的 Isaac Lab 要求 `starlette==0.49.1`，而 `isaaclab_rl` 要求 `packaging<24`。不要单独升级或降级一个包就假定修好了整套环境。

本文不修改旧服务器，也不声称新环境一定能用一次 pip 安装得到零告警。新机应记录 `pip check` 的实际输出；安装失败、新增冲突或冒烟失败均应先处理，不能直接开始正式训练。消除这四项历史冲突属于独立兼容性验证工作。

## 3. 新服务器前置条件

- 推荐 Ubuntu 22.04 x86_64，以贴近原训练主机；Isaac Sim 5.1 官方也列出 Ubuntu 24.04。Ubuntu 20.04 的默认 glibc 不符合此 pip 安装路径。
- Python 必须使用 3.11；Isaac Sim 5.1 的 Linux wheel 要求 glibc >=2.35。
- 两块 RTX 4090 应在 `nvidia-smi` 中正常显示，每块约 24 GB 显存。建议至少 64 GB 主机内存，双进程大规模仿真更适合 128 GB；这些是迁移建议，不是已测出的双卡最低配置。
- 使用持久化 SSD/NVMe，建议预留至少 200 GB 可用空间供环境、Kit 缓存、资产和日志；长期训练建议更大容量。不要放在易清理的容器 overlay 或临时目录。
- 安装支持 RTX 4090、Isaac Sim 5.1 和 CUDA 12.8 的驱动。可参考原驱动 `580.82.07`，并运行官方 Compatibility Checker；服务器驱动由管理员统一维护。
- `nvidia-smi` 的 `CUDA Version` 表示驱动支持能力，不代表安装了对应 CUDA Toolkit。预编译 PyTorch cu128 wheel 带有 CUDA 运行库，常规训练不必额外安装系统 CUDA Toolkit；编译自定义 CUDA 扩展时另行核对工具链。
- 需要能访问 GitHub、PyPI、PyTorch CUDA wheel 源、NVIDIA PyPI 和 Isaac 扩展/资产服务。GPU 可用不代表这些网络访问已经可用。

在新服务器检查：

```bash
uname -m
ldd --version
nvidia-smi
nvidia-smi topo -m
free -h
df -h
```

Ubuntu 22.04 系统依赖示例，由有权限的用户执行；无 sudo 权限时交给管理员。其他发行版或 Ubuntu 24.04 可能使用不同包名，例如部分库有 `t64` 后缀。

```bash
sudo apt-get update
sudo apt-get install -y \
	git git-lfs curl ca-certificates bzip2 unzip rsync tmux \
	build-essential pkg-config ffmpeg \
	libgl1 libegl1 libglfw3 libglib2.0-0 libvulkan1 vulkan-tools \
	libx11-6 libxext6 libxi6 libxrandr2 libxinerama1 libxcursor1 \
	libsm6 libice6 libnss3 libasound2
```

`pynput` 在 Linux 上的 `evdev` 依赖可能需要编译器；原本地环境曾因缺少 gcc 安装失败。下面的 Micromamba 环境也安装 `c-compiler` 作为环境内工具链。

## 4. 目录、源码与 Git LFS

以下命令均是供新服务器执行的迁移步骤，本文没有在新服务器上执行安装。`/data/go2` 是示例路径，应先换成新服务器上有写权限的持久化目录。所有新终端和 tmux 会话都需要设置同样的变量。

```bash
export GO2_ROOT=/data/go2
export PROJECT_DIR="$GO2_ROOT/GO2-ARX5-MUJOCO"
export ISAACLAB_DIR="$GO2_ROOT/IsaacLab"
export MAMBA_ROOT_PREFIX="$GO2_ROOT/mamba"
export PATH="$GO2_ROOT/bin:$PATH"
mkdir -p "$GO2_ROOT/bin" "$GO2_ROOT/envs" "$GO2_ROOT/cache"
```

已有 Micromamba 时复用其可执行文件即可；没有时可用官方二进制入口安装，并核对来源：

```bash
curl --fail --location https://micro.mamba.pm/api/micromamba/linux-64/latest \
	| tar -xj -C "$GO2_ROOT" bin/micromamba
micromamba --version
```

项目与 Isaac Lab 保持为相邻目录，不要把项目复制到 Isaac Lab 源码内部。在新目录克隆：

```bash
git lfs install
git clone --branch feature/reconfigurable-navigation-oracle \
	https://github.com/WM4N-Group/GO2-ARX5-MUJOCO.git "$PROJECT_DIR"
cd "$PROJECT_DIR"
git lfs pull
git merge-base --is-ancestor 629e666eb3479a39069f6443bce493d0ffbb5b65 HEAD
git status --short --branch
```

`merge-base` 返回 0 表示克隆版本包含已验证的集成提交；需要完全固定代码时，可在这个新克隆中 checkout 上述完整 hash。后续已有工作区使用 `git pull --ff-only`，不要覆盖未提交的服务器修改。

[.gitattributes](../.gitattributes) 为 USD、网格、策略等声明了 LFS。必须检查资产内容，不能仅确认文件存在：LFS 指针也是一个存在的文本文件。若 `git lfs pull` 报对象不存在或权限问题，先恢复远端 LFS 对象或从已验证机器同步完整资产，不能带着指针文件进入仿真。

2026-09-11 已额外核对远端同步提交 `629e666`：下面三个 GO2 资产目录中的策略、USD 子层和网格实际存储为完整 Git blob，而非 LFS 指针，因此这些已核对的关键文件随普通 `git clone` 即可取得。LFS 声明不等于文件已经转换为 LFS 对象；新机若提示文件本应为指针，应先核对实际内容和校验和，不要为消除提示而重写仓库历史。此检查不覆盖其他机器人或未来新增的 LFS 资产。

迁移必须保留以下目录及其引用的文件：

- [MuJoCo 机器人 XML 与网格](../mujoco/robots/go2_arx5/)。
- [Isaac GO2-ARX5 USD 与 configuration 子层](../source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_arx5/)。
- [NAV 与 CLIMB 部署策略](../mujoco/deploy/policy/go2_arx5/)。
- [可重构导航场景与运行时](../mujoco/reconfigurable_navigation/)。

不要只复制 `go2_arx5.usd` 而漏掉其子层，也不要只复制 `.py`。机器人 USD 在本仓库中；其他 Isaac 默认材质/扩展仍可能首次联网下载。

## 5. 安装 MuJoCo 推理环境

```bash
micromamba create -y -p "$GO2_ROOT/envs/go2-mujoco" -c conda-forge \
	python=3.11.16 pip c-compiler
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$GO2_ROOT/envs/go2-mujoco"
cd "$PROJECT_DIR"

python -m pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt \
	"mujoco==3.12.0" "numpy==2.4.6" "scipy==1.17.1" \
	"PyYAML==6.0.3" "pynput==1.8.2" "psutil==7.2.2" "prettytable==3.18.0"
python -m pip install -e source/LeggedManip_Lab
python -m pip check
```

项目扩展的 build-system 已声明 `setuptools<82.0.0`、`wheel`、`toml`，正常 pip editable 安装会使用这些构建依赖。此环境不需要安装 Isaac Lab，直接执行 `scripts/list_envs.py` 则会因为缺少 Isaac 栈而失败，这是预期的环境边界。

仅需要可选 MuJoCo PPO 训练时，再执行以下命令；它不是复现现有 CLIMB 的训练入口：

```bash
python -m pip install -r mujoco/train/go2_arx5/requirements.txt
```

### 5.1 策略校验和

在仓库根目录运行：

```bash
sha256sum mujoco/deploy/policy/go2_arx5/policy.pt \
	mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt
```

当前已验证值：

```text
d46a829f8cc2f85f19a030140092a206880e56f42aca6314be4e165e847de347  mujoco/deploy/policy/go2_arx5/policy.pt
ef88741365e85365486c30762bd792e57292eb8798b5f5aaa19af2ebb8495af5  mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt
```

中间策略 `policy_iter348.pt`、`policy_iter900.pt` 不属于必需迁移文件；最终策略已经纳入 `629e666`。不要因迁移而重新训练或替换 actor。

### 5.2 无头验收

```bash
python mujoco/check_reconfigurable_navigation.py --seeds 100
python mujoco/evaluate_policy_navigation.py --seeds 100
python mujoco/check_push_skill.py --seeds 20
python mujoco/check_climb_skill_switching.py --seeds 10
python mujoco/check_complex_course.py --seeds 10
python mujoco/visualize_climb_policy.py \
	--policy mujoco/deploy/policy/go2_arx5/climb/policy_iter1499.pt \
	--duration 20 --headless --no-realtime
```

原验收记录分别为 Oracle `100/100`、NAV `100/100`、PUSH `20/20`、CLIMB 切换 `10/10`、复杂课程 `10/10`。新机应重新记录结果，不把旧数字直接当成新机结果。复杂课程成功还应包含 `reason=goal_reached`、`resets=1`、指尖接触成立、无机身接触与非法碰撞。

上述物理检查不需要 X11，也不需要创建图像渲染器。交互 `--visualize` 需要真实可用的 DISPLAY/OpenGL；SSH 终端、`MUJOCO_GL=egl` 或 Xvfb 本身不等价于可交互桌面。需要 MuJoCo 离屏渲染时才考虑 EGL；远程查看优先录制视频后下载，或配置受支持的远程桌面。

## 6. 安装 Isaac 训练环境

只运行现成的 MuJoCo 复杂课程时，可以跳过本节。

### 6.1 创建环境并固定基础组件

```bash
micromamba create -y -p "$GO2_ROOT/envs/go2-isaac" -c conda-forge \
	python=3.11.16 pip c-compiler 'libstdcxx-ng>=14'
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$GO2_ROOT/envs/go2-isaac"
python -m pip install --upgrade pip 'setuptools<82' wheel toml
python -m pip install torch==2.7.0 torchvision==0.22.0 \
	--index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
	--extra-index-url https://pypi.nvidia.com
```

官方安装选择器写作 `5.1.0`，安装后的包元数据为 `5.1.0.0`。`extscache` 会增加下载和磁盘占用，但可减少首次启动时动态下载 Kit 扩展的需求，不代表所有场景资产都已离线可用。

使用 Isaac Sim 前须阅读并接受 NVIDIA Omniverse EULA。仅在你同意条款后，为当前终端设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
```

### 6.2 安装固定提交的 Isaac Lab 和项目

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
git -C "$ISAACLAB_DIR" checkout b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
cd "$ISAACLAB_DIR"
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
	./isaaclab.sh --install rsl_rl

cd "$PROJECT_DIR"
python -m pip install -r requirements.txt \
	'mujoco==3.12.0' 'numpy==1.26.0' 'scipy==1.15.3' \
	'torch==2.7.0' 'torchvision==0.22.0' 'gymnasium==1.2.1' \
	'rsl-rl-lib==5.0.1' 'tensordict==0.14.1' 'prettytable==3.3.0'
python -m pip install -e source/LeggedManip_Lab
python -m pip show isaacsim isaaclab isaaclab_rl rsl-rl-lib torch torchvision numpy
python -m pip check
```

锁定的 Isaac Lab 安装器负责安装自身扩展和 RSL-RL extra。仍有传递依赖使用宽范围，以上命令不是完整锁文件；安装后必须确认 PyTorch 仍为 `2.7.0+cu128`，而不是 CPU 版或意外升级的版本。不要在此环境执行前节的 CPU PyTorch 安装命令。

`pip check` 的已知历史告警见 2.2 节。旧机器曾在安装末尾生成 VS Code 配置的步骤挂起，而核心包已经安装完成；如果再次遇到此情况，先检查具体进程和包元数据，不要误停仍在安装的 pip，也不要从头重复下载整个 Isaac 栈。

### 6.3 新服务器启动封装

旧启动器位于仓库外，且硬编码旧服务器的环境路径，不能直接原样使用。下面在新服务器生成一个使用 `GO2_ROOT` 的封装。它只对子进程预加载环境内的 libstdc++，不替换系统库，也不代替用户接受 EULA。

```bash
cat > "$GO2_ROOT/bin/isaac-python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${GO2_ROOT:?Set GO2_ROOT to the persistent project root}"
env_prefix="$GO2_ROOT/envs/go2-isaac"
export PATH="$env_prefix/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_PRELOAD="$env_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$env_prefix/bin/python" "$@"
SH
chmod +x "$GO2_ROOT/bin/isaac-python"
```

原服务器如果不预加载环境里的 libstdc++，Kit 会选择系统库并报 `CXXABI_1.3.15`。新机库路径不存在时，应检查环境创建结果，不应直接替换 `/usr/lib` 下的系统文件。不要把此预加载设置写成影响整台机器的全局设置。

### 6.4 单卡最小训练验收

先确认两块卡均对 CUDA 可见，再逐卡做小规模测试。以下命令会产生新的迁移冒烟日志，不是启动正式 CLIMB 重训。

```bash
isaac-python -c 'import torch; print(torch.__version__, torch.version.cuda); print("GPU count:", torch.cuda.device_count()); print([torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]); assert torch.cuda.is_available()'
cd "$PROJECT_DIR"
isaac-python scripts/list_envs.py
CUDA_VISIBLE_DEVICES=0 isaac-python scripts/rsl_rl/train.py \
	--task GO2-ARX5-Climb --device cuda:0 --num_envs 16 \
	--max_iterations 1 --headless --logger tensorboard \
	--experiment_name go2_migration_smoke --run_name gpu0
CUDA_VISIBLE_DEVICES=1 isaac-python scripts/rsl_rl/train.py \
	--task GO2-ARX5-Climb --device cuda:0 --num_envs 16 \
	--max_iterations 1 --headless --logger tensorboard \
	--experiment_name go2_migration_smoke --run_name gpu1
```

当 `CUDA_VISIBLE_DEVICES=1` 时，进程内 `cuda:0` 映射到物理 GPU 1，不要同时写成 `--device cuda:1`。任务列表应包含 `GO2-ARX5-Flat{,-Play}`、`GO2-ARX5-WBC{,-Play}`、`GO2-ARX5-Climb{,-Play}`。任务注册成功仍不等于资产加载和训练成功；每次冒烟应实际完成一次迭代并生成 `model_0.pt`。

## 7. 双 RTX 4090 的使用方式

两块 4090 不是单块 48 GB 显卡。4090 无 NVLink，每个训练进程仍受所用单卡的显存限制；两卡通信经过 PCIe 等主机通路，速度还取决于主板拓扑和 CPU。多卡不会自动提高此项目的训练速度，也不会自动合并显存。

### 7.1 推荐起步：每卡一个独立任务

单卡冒烟通过后，可在两个独立 tmux 会话中运行不同 seed 的训练，或者一卡训练、另一卡评估。两进程不是同一次分布式训练，日志名必须不同。

在各会话中重新设置第 4 节变量、激活环境并按已接受的许可状态设置 EULA。确认确实需要新训练后，再运行以下示例：

```bash
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=0 isaac-python scripts/rsl_rl/train.py \
	--task GO2-ARX5-Climb --device cuda:0 --num_envs 4096 \
	--seed 0 --max_iterations 1500 --headless --logger tensorboard \
	--run_name dual_host_gpu0_seed0
```

另一个 tmux 会话：

```bash
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=1 isaac-python scripts/rsl_rl/train.py \
	--task GO2-ARX5-Climb --device cuda:0 --num_envs 4096 \
	--seed 1 --max_iterations 1500 --headless --logger tensorboard \
	--run_name dual_host_gpu1_seed1
```

单卡 4096 环境已有历史记录，但双任务同时运行的 CPU、RAM 和散热需求仍需测量。先以 16/256 环境试运行，再放大到 4096。使用 `nvidia-smi` 核对进程和显存；涉及相机/Vulkan 渲染时，还应检查 Kit 的渲染设备选择，不能只凭 CUDA 掩码断言全部渲染负载都已隔离。

### 7.2 可选：同一训练任务使用两块卡

[训练入口](../scripts/rsl_rl/train.py) 已有 `--distributed` 和 `local_rank` 设备设置。正确方式是一个 launcher 启动两个进程，而不只是设置 `CUDA_VISIBLE_DEVICES=0,1`。

以下为基于当前代码接口的双卡冒烟命令，尚未在新服务器实测：

```bash
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=0,1 isaac-python -m torch.distributed.run \
	--standalone --nnodes=1 --nproc_per_node=2 \
	scripts/rsl_rl/train.py --distributed \
	--task GO2-ARX5-Climb --num_envs 16 --max_iterations 1 \
	--headless --logger tensorboard \
	--experiment_name go2_migration_ddp_smoke --run_name two_gpu
```

- 当前代码把 `--num_envs` 传给每个进程的环境配置，没有自动除以 world size：上述命令共 32 个环境。
- 若希望总环境数与旧单卡 4096 相同，双卡应从每进程 `--num_envs 2048` 开始，而不是直接每卡 4096。
- 正式训练前核对 NCCL 初始化、两个 rank 的 GPU 分配、环境数、NaN、checkpoint 和日志路径。旧单卡通过不能代替双卡通过。
- 不同 global batch、随机种子和通信顺序会影响训练轨迹；双卡加速不是逐步数值等价的保证。
- 遇到 NCCL 卡住，先检查 `nvidia-smi topo -m`、容器共享内存和 GPU 可见性。可用 `NCCL_DEBUG=INFO` 获取诊断；不要默认关闭 P2P/IB，或修改全局 NCCL 配置。
- 不要同时把两卡分配给两个独立满负载任务和一个双卡 DDP 任务。

## 8. Checkpoint、日志与续训迁移

Git 中的 TorchScript actor 足以执行 MuJoCo，但不能恢复 RSL-RL 的 critic、优化器和训练进度。要在新服务器续训，必须另外迁移原始 checkpoint。

| 内容 | 是否必须 | 来源/去向 |
| --- | --- | --- |
| 项目源码、XML/USD/mesh、最终部署 actor | 必须 | Git + LFS + 策略校验和 |
| CLIMB `model_1499.pt` | 续训/原生 checkpoint 评估必需 | 旧服务器最终训练 run |
| `params/env.yaml`、`params/agent.yaml` | 强烈建议一并保留 | 与 checkpoint 同一 run，用于核对训练配置 |
| TensorBoard events、exported、视频 | 按研究记录需要迁移 | 同一 run 或原本地 artifacts |
| 两个中间 actor | 非必需 | 仅诊断用途 |
| `.envs`、Conda/Micromamba 环境目录 | 不建议直接复制 | 在新路径重建，避免绝对前缀、动态库和 editable 路径失效 |
| `.ssh` 私钥、令牌、LiveAgent 配置库 | 不属于项目依赖 | 不写入仓库；新服务器独立配置访问权限 |

旧训练 run 的实际路径是：

```text
/mnt/miaojigui/worktrees/go2-climb/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/
```

在新服务器配置好对旧主机的 SSH 访问后，可同步整个 run；`old-4090` 是占位 SSH 别名，不是新机自动具备的配置。没有直连条件时，经可信开发机中转同一目录。

```bash
export OLD_TRAIN_HOST=old-4090
mkdir -p "$PROJECT_DIR/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49"
rsync -avh --progress \
	"$OLD_TRAIN_HOST:/mnt/miaojigui/worktrees/go2-climb/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/" \
	"$PROJECT_DIR/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/"
```

迁移前后对 `model_1499.pt` 运行 `sha256sum` 并比较。ONNX 和跟随录像原本另存于开发机的 `/home/yuanyue/re-nav/artifacts/go2-climb/iter1499/`，不保证仅克隆仓库就会得到它们。

先进行有限步原生评估与导出，确认 checkpoint、仿真和录像链路：

```bash
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=0 isaac-python scripts/rsl_rl/play.py \
	--task GO2-ARX5-Climb-Play --device cuda:0 --num_envs 16 \
	--checkpoint "$PROJECT_DIR/logs/rsl_rl/go2_arx5_climb/2026-09-10_15-03-49/model_1499.pt" \
	--headless --video --video_length 1000
```

当前 `play.py` 会自动导出 JIT 和 ONNX，并在录完上述 1000 步后退出；不带 `--video` 的 headless play 通常持续运行，需要主动停止。录像需要相机渲染及编码依赖，不能等同于纯物理无头训练。

确认需要继续优化已有策略时，续训示例：

```bash
CUDA_VISIBLE_DEVICES=0 isaac-python scripts/rsl_rl/train.py \
	--task GO2-ARX5-Climb --device cuda:0 --num_envs 4096 \
	--resume --load_run 2026-09-10_15-03-49 --checkpoint model_1499.pt \
	--max_iterations 100 --headless --logger tensorboard \
	--run_name resumed_on_new_server
```

这里训练入口按 `logs/rsl_rl/go2_arx5_climb/<load_run>/<checkpoint>` 查找文件，而 play 的 `--checkpoint` 支持直接给路径。`--max_iterations` 直接传给 runner 的 `num_learning_iterations`；恢复时不要把它误认为全局停止迭代号，先用 `1` 做恢复测试并检查打印出的迭代区间。旧 run 的 YAML 不会替代当前代码中的任务配置，续训前需要比对它们。

## 9. 常见迁移故障

| 症状 | 优先检查 |
| --- | --- |
| 找不到 isaacsim wheel | Python 是否为 3.11、glibc 是否 >=2.35、CPU 架构及 NVIDIA 包源是否可访问 |
| `CXXABI_1.3.15` 或 libstdc++ 导入失败 | 是否使用新路径的启动封装；环境内 libstdc++ 是否存在 |
| `RslRlMLPModelCfg` 等导入失败 | 是否安装了固定 Isaac Lab commit 和 RSL-RL 5.0.1；是否误用旧 release |
| `torch.cuda.is_available()` 为 False | 是否误装 CPU wheel、驱动/容器 GPU 透传、CUDA_VISIBLE_DEVICES 设置 |
| USD/mesh 无法解析、策略无法加载 | Git LFS 是否下载真实内容、USD 子层是否完整、hash 是否一致 |
| `pynput`/X11/DISPLAY 错误 | 使用无头检查入口，不要在无图形会话强行导入键盘遥操作组件 |
| Isaac 首次启动很慢 | 扩展/资产下载与 shader 缓存；检查日志、网络及磁盘，不把首次慢启动直接当成死锁 |
| Git HTTP 408 | 排查网络/代理上传链路；本次功能提交约 1.1 MiB，不属于超大文件限制问题 |
| 双卡一张空闲或另一张 OOM | 每进程设备映射、实际环境数、Vulkan/Kit 设备选择、是否同时跑了其他作业 |
| MuJoCo CLIMB 再次跌倒 | 先核对版本、最终 actor、模型与 runtime；不通过改增益或重训掩盖迁移问题 |

不要对代理 URL、凭据、私钥做公开环境快照。需要锁定旧环境时，可另外记录 `pip list --format=freeze`、Micromamba 包列表、Isaac Lab commit、驱动信息和模型 hash；导出的完整环境清单仍需审查本地路径和私有下载源，不能直接当成可移植的 requirements 文件。

## 10. 迁移完成清单与参考

- [ ] 新服务器两块 GPU、驱动、glibc、RAM 和持久化磁盘已确认。
- [ ] 代码包含 `629e666`；Isaac Lab 固定为本文记录的完整 commit。
- [ ] Git LFS 资产完整；NAV 和 CLIMB 策略 hash 与本文一致。
- [ ] 两个 Python 环境分离；NumPy、PyTorch CPU/CUDA 版本没有串用。
- [ ] `pip check` 输出已记录并审查，新增冲突未被忽略。
- [ ] MuJoCo 规划与三技能物理回归在新机重新通过。
- [ ] GPU 0 和 GPU 1 的 Isaac 单卡冒烟分别完成并产出 checkpoint。
- [ ] 原始 CLIMB checkpoint、训练 YAML 和必要日志已迁移并校验。
- [ ] 若启用 DDP，双卡冒烟与实际加速收益已单独验证。
- [ ] 长任务放入 tmux；TensorBoard 仅绑定回环地址并通过 SSH 转发访问。

本文已核对本地包元数据、旧单卡服务器 commit/包元数据及 `pip check`、当前训练脚本参数、策略 SHA-256 和官方安装要求。新服务器安装、双卡并行和 DDP 仍待在目标硬件上执行，本文不是新服务器已经通过验收的报告。

参考：

- [Isaac Sim 5.1 Python 安装与 EULA](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [固定的 Isaac Lab 提交](https://github.com/isaac-sim/IsaacLab/tree/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8)
- [项目安装说明](../README_CN.md)
- [可重构导航运行与验收说明](../mujoco/reconfigurable_navigation/README_CN.md)
- [CLIMB 训练与导出记录](CLIMB_TRAINING_CN.md)

官方 5.1 文档目前标注该版本已不再受支持。这里固定旧版本是为了复现项目已经验证的行为；未来升级 Isaac Sim/Isaac Lab 应使用独立环境和回归，不与服务器迁移同时进行。