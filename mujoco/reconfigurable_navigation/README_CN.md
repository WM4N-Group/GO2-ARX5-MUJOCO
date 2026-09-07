# GO2-ARX5 可重构导航原型

该目录实现世界模型项目前置的 Oracle-first 原型。当前阶段使用 MuJoCo 真值状态，不依赖 RGB-D、VLM 或学习世界模型，用于验证场景定义、结构化技能接口和可达性判断。

## 当前功能

- GO2-ARX5 Blocked Passage 专用 MuJoCo 场景。
- 带自由关节、质量和摩擦的动态箱体。
- 结构化 `Observation`、`Capability`、`ObjectState` 和 `SkillAction`。
- 机器人 footprint 膨胀的二维占据栅格。
- 支持八邻域、禁止穿角的 A* 路径规划。
- 通过移除单个可移动物体识别阻挡物。
- Oracle 生成 `NAV -> PUSH -> NAV -> STOP` 技能序列。
- 根据 `max_pushable_mass` 拒绝能力不足的方案。
- 统一 Skill 生命周期和 NAV 目标位姿控制器。
- 将世界坐标误差转换为策略使用的机体系 `vx/vy/yaw_rate`。

## 运行自检

使用安装了 MuJoCo 的 `leggedmanip` 环境：

```bash
conda run -n leggedmanip python mujoco/check_reconfigurable_navigation.py
```

指定随机场景数量：

```bash
conda run -n leggedmanip python mujoco/check_reconfigurable_navigation.py --seeds 500
```

检查内容包括：

1. 箱体位于通道中时，目标不可直接到达。
2. 规划器正确识别阻挡箱体 `object_id=10`。
3. 生成 `NAV -> PUSH -> NAV -> STOP` 结构化计划。
4. Oracle 设置推箱结果后，目标变为可达。
5. 箱体质量超过 Capability 时，规划器拒绝执行。
6. NAV 输出满足速度上限，并在到达目标位姿后停止。

## Oracle 规划可视化

循环显示随机场景：

```bash
conda run -n leggedmanip python mujoco/visualize_reconfigurable_navigation.py
```

只播放一次，或提高动画速度：

```bash
conda run -n leggedmanip python mujoco/visualize_reconfigurable_navigation.py --once
conda run -n leggedmanip python mujoco/visualize_reconfigurable_navigation.py --speed 2
```

Viewer 将依次显示机器人接近箱体、将橙色箱体推出阻挡区域、重新规划并到达绿色目标点。终端同步打印当前规划和执行阶段。关闭 Viewer 窗口即可停止循环。

该动画是 Oracle 规划和场景重构的运动学预览，机器人与箱体位姿由演示器逐帧设置；它不表示底层策略已经完成真实接触推动。

## 真实动力学 NAV 可视化

以下入口不再修改机器人基座位姿。它构造 210 维历史观测，运行现有 `210 -> 18` TorchScript 策略，通过 PD 力矩控制和 `mujoco.mj_step()` 产生真实步态：

```bash
conda run -n leggedmanip python mujoco/visualize_policy_navigation.py
```

无 Viewer 快速诊断：

```bash
conda run -n leggedmanip python mujoco/visualize_policy_navigation.py \
  --headless --no-realtime
```

当前默认部署策略可以产生明显步态和前进运动，但在原始平地环境和重构场景中都会很快失稳。本地 `go2_arx5_visual` 策略可以稳定站立，但只训练了约一万步，尚未学会速度跟踪。因此，真实 NAV 入口目前是控制链路诊断工具，还不能完成目标点导航。必须先修复训练配置并得到可用 locomotion checkpoint，才能继续实现物理 PUSH。

## 模块

| 文件 | 作用 |
| --- | --- |
| `representations.py` | 观测、对象、能力和技能动作类型 |
| `env.py` | MuJoCo 场景加载、随机化与真值观测 |
| `occupancy.py` | 占据栅格、障碍膨胀和 A* |
| `oracle_planner.py` | 阻挡物识别与规则式技能规划 |
| `locomotion_runtime.py` | 210 维观测、TorchScript 推理、PD 控制和 MuJoCo 动力学 |
| `skills/base.py` | 统一技能生命周期和低层命令类型 |
| `skills/navigate.py` | NAV 目标跟踪和速度命令生成 |
| `blocked_passage.xml` | Blocked Passage 场景 |

## 当前边界

`set_box_pose()` 只用于验证反事实重构结果，尚未模拟真实 PUSH 技能。NAV 控制器和真实动力学运行时已经接入，但现有 checkpoint 不满足稳定速度跟踪要求。下一阶段必须先修复 locomotion 训练，再实现箱体接近、接触、推动和退出状态机。
