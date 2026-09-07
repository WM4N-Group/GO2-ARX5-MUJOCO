# GO2-ARX5 世界模型可重构导航 Demo 设计与实施计划

## 1. 项目背景

传统机器人导航通常将环境视为固定不变：机器人只能在已有自由空间中寻找路径，并通过避障、跨越或绕行到达目标。然而在真实室内环境中，可行路径并不总是预先存在：通道可能被物体堵塞，高台可能超过机器人攀爬高度，沟槽也可能超过其最大跨越距离。

本项目希望赋予 GO2-ARX5 主动改变环境的能力。当目标不可达时，机器人能够识别和利用环境中的可移动物体，通过推动、借助支撑物攀爬或修复通路，将原本不可通行的场景重构为可通行场景。

环境的“可通行性”不是固定属性，而取决于机器人的形态和技能库。对于只会平地行走的机器人，一处障碍可能完全不可通过；对于具有攀爬、跳跃或环境交互能力的机器人，同一场景可能是可达的。因此，系统还需要评估不同机器人能力对可达空间和重构方案的影响。

## 2. 项目目标

GO2-ARX5 根据 RGB-D 场景观测、语义几何地图、机器人能力和任务目标，在常规导航不可达时，主动移动或利用环境中的物体，使目标变得可达，并通过世界模型预测和闭环重规划完成任务。

系统采用以下职责划分：

- VLM 决定 **WHAT**：提出环境改造方案和技能骨架候选。
- 几何模块决定 **WHERE**：将语义方案落到具体对象、支撑物和目标位姿。
- 世界模型决定 **WHETHER**：预测技能执行后的状态、成功概率、风险和后续可达性。
- MuJoCo 或真实机器人负责执行：每个技能结束后重新观测、更新状态并规划。

本项目中的世界模型是对象中心、技能级、可用于规划搜索的动力学模型，而不是通用视频生成模型。

## 3. Demo 范围

### 3.1 MVP：Blocked Passage

场景中的通路被可移动箱体堵住。系统需要完成：

```text
NAV 到箱体前方
-> PUSH 箱体到通路外的目标位姿
-> NAV 到最终目标
-> STOP
```

MVP 应完整展示：

- 常规导航判断目标不可达。
- 识别阻挡通路的可移动物体。
- 生成环境重构候选方案。
- 世界模型预测箱体移动结果、成功率和碰撞风险。
- 执行最优重构方案。
- 重构后恢复路径可达性。
- 技能失败或状态偏差时重新观测和规划。

### 3.2 第二阶段：High Platform Access

```text
NAV 到箱体
-> PUSH 箱体到高台前
-> CLIMB 到箱体
-> CLIMB 到高台
-> NAV 到目标
```

该任务重点验证“机器人能力决定可达性”。改变 `max_step_height`、`max_pushable_mass` 等 Capability 后，规划结果应发生合理变化。

### 3.3 第三阶段：Gap Crossing / Gap Repair

- 能力足够时直接执行 `JUMP`。
- 能力不足且存在合适物体时，通过 `PUSH/PLACE -> CLIMB/JUMP` 修复通路。
- 没有可行支撑物时返回不可达。

搭桥任务涉及长物体接触、姿态控制和支撑稳定性，应在推箱与攀爬技能稳定后实现。

## 4. 总体架构

```text
MuJoCo / 真实 RGB-D
        |
        v
感知与几何重建
        |
        v
Semantic BEV + Object Tokens + Robot State
        |
        v
技能库与 Capability 约束
        |
        v
VLM 提出技能骨架候选
        |
        v
几何 Grounding 生成对象、支撑物和目标位姿
        |
        v
世界模型预测技能级状态转移、成功率和风险
        |
        v
技能图搜索 + CEM 连续参数优化
        |
        v
执行最优计划的第一个技能
        |
        v
重新观测、对象关联、失败检测与重规划
```

VLM 不直接输出电机动作，也不作为物理可行性的最终判断器。即使 VLM 暂时不可用，系统也必须能使用模板候选生成器运行完整闭环。

## 5. 数据接口

### 5.1 观测定义

每个技能执行前的观测定义为：

```text
O_t = {
  rgbd_views,
  camera_intrinsics,
  camera_extrinsics,
  semantic_bev,
  object_tokens,
  robot_state,
  goal,
  capability
}
```

### 5.2 Semantic BEV

沿用原始设计规格：

```text
shape: 8 x 128 x 128
resolution: 0.05 m
coverage: 6.4 m x 6.4 m
```

建议固定 8 个通道：

1. 占据概率。
2. 最大高度。
3. 地面坡度或粗糙度。
4. 可支撑概率。
5. 可移动概率。
6. 静态障碍语义。
7. 已观测区域和重建置信度。
8. 机器人与目标的空间提示。

目标和机器人状态仍应单独编码，第 8 通道只提供空间位置提示。

### 5.3 Object Tokens

沿用 `32 x 16` 的对象表示：

| 维度 | 含义 |
| --- | --- |
| `0:3` | 对象中心 `x, y, z` |
| `3:6` | 对象尺寸 `length, width, height` |
| `6` | 是否可移动 |
| `7` | 是否可支撑 |
| `8:15` | 对象类别 one-hot |
| `15` | 重建置信度和有效节点标记 |

对象类别初期包括：

```text
floor
movable_box
static_obstacle
platform
bridge
goal_marker
other
```

`robot` 和 `goal` 不作为普通对象参与动力学预测，而是由独立编码器处理。

### 5.4 Robot State

固定为 12 维：

```text
x, y, z,
roll, pitch, yaw,
vx, vy, vz,
yaw_rate,
support_foot_count,
state_valid
```

对象与目标的相对位置不放入 Robot State，因为场景中可能存在多个可操作对象。相对关系由 Object Tokens、Goal 和空间编码器处理。

### 5.5 Goal

目标定义为：

```text
goal = [goal_x, goal_y, goal_z, valid]
```

对于平面导航任务，`goal_z` 为目标平面高度；对于高台任务，它表示目标平台高度。

### 5.6 Capability

使用 8 维能力向量：

```text
body_length
body_width
max_step_height
max_gap_width
max_slope
max_push_force / 100
max_pushable_mass
max_linear_speed
```

Capability 必须参与数据随机化。同一场景应使用不同能力向量生成不同可达性标签，使模型真正学习能力相关的规划结果。

### 5.7 高层动作

采用结构化技能动作：

```text
A_t = {
  skill,
  object_pointer,
  support_pointer,
  parameters
}
```

技能集合：

```text
NAV
PUSH
CLIMB
JUMP
STOP
```

参数定义：

| Skill | Parameters |
| --- | --- |
| `NAV` | `target_pose[x, y, yaw]` |
| `PUSH` | `object_target_pose[x, y, yaw]` |
| `CLIMB` | `landing_pose[x, y, z, yaw]` |
| `JUMP` | `landing_pose[x, y, z, yaw]` |
| `STOP` | 空参数 |

模型内部可将动作编码为固定长度 token，但模块外部接口不暴露通用 53 维数组。

### 5.8 监督输出

```text
Y_t = {
  next_robot_state,
  next_selected_object_state,
  skill_success,
  collision,
  support_stable,
  terminal,
  task_success,
  reachability,
  execution_cost
}
```

单条训练数据定义为：

```text
D_t = (O_t, A_t, O_{t+1}, Y_t)
```

数据集中必须同时包含成功动作、失败动作、边界动作和反事实候选，避免成功概率头只看到可行动作。

## 6. 世界模型

### 6.1 编码器

- BEV Encoder：3 到 4 层 CNN，输出 128 或 256 维空间特征。
- Object Encoder：Object Transformer，保留各对象 token，不提前做简单均值池化。
- State Encoder：分别编码 Robot、Goal 和 Capability。
- Action Encoder：编码技能类型、对象指针、支撑指针和连续参数。

### 6.2 Dynamics Core

采用 4 到 6 层 Transformer。输入 token 序列为：

```text
[scene token]
[robot token]
[goal token]
[capability token]
[object tokens x N]
[action token]
```

输出预测头：

- `delta_robot_state`
- `delta_selected_object_state`
- `skill_success`
- `collision`
- `support_stable`
- `terminal`
- `reachability/value`

建议训练 3 到 5 个独立模型组成 ensemble。规划时将模型之间的预测分歧作为 epistemic uncertainty，降低模型在分布外场景中过度自信的风险。

### 6.3 损失函数

```text
L =
  lambda_robot * Huber(next_robot)
+ lambda_object * Huber(next_object)
+ lambda_success * BCE(success)
+ lambda_collision * FocalLoss(collision)
+ lambda_support * BCE(support_stable)
+ lambda_terminal * BCE(terminal)
+ lambda_reach * BCE(reachability)
+ lambda_value * Huber(cost_to_goal)
```

`reachability` 应来自完整 episode 的回溯标签，或由后续 Oracle 规划能否到达目标生成，不能只使用单步标签。

## 7. 规划方法

第一版采用“离散技能搜索 + 连续参数 CEM”，不采用纯端到端动作生成。

1. VLM 或模板生成器提出 3 到 10 个技能骨架。
2. 几何模块过滤不合法的对象和支撑物指针。
3. 对每个骨架使用 CEM 优化 PUSH 目标位姿和 landing pose 等连续参数。
4. 世界模型对候选方案 rollout 3 到 5 个技能。
5. 综合任务成功率、目标距离、碰撞、支撑风险、执行成本和模型不确定度进行排名。
6. 只执行最优计划中的第一个技能。
7. 技能结束、超时或异常发生后重新观测并规划。

候选评分函数：

```text
score =
  w_goal * P(task_success)
+ w_reach * reachability
- w_collision * P(collision)
- w_support * P(unstable)
- w_cost * execution_cost
- w_uncertainty * ensemble_disagreement
```

世界模型置信度不足时，系统应执行重新观测、保守动作或拒绝执行。

## 8. 技能库

当前 GO2-ARX5 MuJoCo 环境已具备：

- 平地速度指令。
- 机械臂末端位姿指令。
- 210 维本体观测。
- 18 维关节动作。
- PPO 训练和 TorchScript 部署导出。

现有能力还不等于完整技能库，需要新增以下模块。

### 8.1 NAV

输入目标位姿 `x, y, yaw`，输出速度指令给现有步态策略。第一版使用几何路径规划器和 pure-pursuit/SE(2) 控制器，不必重新训练导航策略。

### 8.2 PUSH

第一版使用状态机实现：

1. 导航到对象接近位姿。
2. 调整机器人和机械臂接触姿态。
3. 沿规划方向推动箱体。
4. 检测箱体目标位姿、接触丢失或异常碰撞。
5. 成功后退出接触区域。

后续可训练更鲁棒的接触策略替代状态机。

### 8.3 CLIMB

需要独立训练台阶和箱体攀爬策略，输入深度图、高度图或高度扫描。现有平地策略不能直接承担该技能。

### 8.4 JUMP

跳跃技能训练风险和硬件风险最高，不应阻塞 MVP。应在 NAV、PUSH 和 CLIMB 稳定后加入。

### 8.5 STOP / Recovery

包括站立、急停、跌倒检测、技能超时和恢复策略。Recovery 是闭环系统的必要组成。

所有技能实现统一接口：

```text
can_execute(observation, parameters)
reset(parameters)
step(observation)
is_success(observation)
is_failed(observation)
timeout
```

## 9. 数据生成

### 9.1 阶段 A：Oracle 数据

先直接读取 MuJoCo 特权状态，验证系统闭环：

- 平台高度、尺寸和位置。
- 箱体尺寸、质量、摩擦和初始位姿。
- 机器人初始位姿和朝向。
- 目标位置。
- 技能参数。
- 接触、碰撞和支撑状态。

场景分布：

```text
25% 直接可达
50% 需要环境重构
25% 无可行方案
```

场景随机化包括：

- 平台高度、长宽和中心位置。
- 箱体高度、长宽、质量、摩擦、初始位置和偏航角。
- 机器人初始位置和朝向。
- 推箱目标位姿。
- 重建位置和尺寸误差。

### 9.2 阶段 B：感知数据

增加 4 到 6 个固定 RGB-D 相机：

- 生成 RGB、Depth、实例分割和相机标定。
- 训练时实例 ID 只用于监督标签。
- 推理时只允许通过 RGB-D 重建。
- MJCF 精确几何不得直接输入推理模型。

感知扰动包括：

- 深度噪声与遮挡。
- 相机外参误差。
- 对象位置和尺寸误差。
- 对象漏检和错误分类。
- 摩擦和质量随机化。
- 机器人状态估计噪声。

### 9.3 阶段 C：Isaac / 真实视觉迁移

使用 Isaac Sim 多视角 RGB-D 和匿名实例 mask 作为域外测试或视觉预训练数据，不将 Isaac Sim 作为 MuJoCo MVP 的硬依赖。

## 10. 闭环执行

系统采用 receding-horizon 执行：

1. 获取当前 RGB-D 和机器人状态。
2. 构建 BEV，并完成对象检测和 ID 关联。
3. 判断是否可直接导航到目标。
4. 不可达时生成重构候选。
5. 使用世界模型与几何约束筛选候选。
6. 执行最优候选的第一个技能。
7. 技能完成或失败后重新观测。
8. 对比预测状态与真实状态。
9. 更新场景图，重新规划直到成功、确定不可达或达到预算上限。

执行器必须处理：

- 对象滑移和推箱偏差。
- 接触丢失。
- 非法碰撞。
- 机器人失稳或跌倒。
- 对象 ID 变化。
- 世界模型预测与真实执行偏差。
- 技能超时和无进展检测。

## 11. 开发里程碑

### P0：冻结接口，2 到 3 天

- 确定 Observation、Action 和 Label schema。
- 确定 MVP 只实现 Blocked Passage。
- 定义成功、碰撞、支撑和可达标签。
- 建立数据版本号和配置文件。

### P1：可重构 MuJoCo 场景，1 周

- 动态生成平台、通道、箱体和沟槽。
- 支持质量、摩擦、尺寸和位姿随机化。
- 实现特权 BEV 和 Object Tokens。
- 实现 RGB-D 与实例标签采集。
- 建立场景和规划可视化。

### P2：NAV + PUSH 技能，2 周

- 复用现有速度跟踪策略。
- 实现目标点导航控制器。
- 实现推箱接近、对齐、推动和退出状态机。
- 建立统一 Skill API。
- 测量单技能成功率和失败模式。

### P3：Oracle 高层规划，1 周

- 不使用学习世界模型，直接通过 MuJoCo rollout 验证候选。
- 跑通 `NAV -> PUSH -> NAV` 完整闭环。
- 验证场景定义和动作参数化。
- 将 Oracle 规划器作为数据教师和性能上界。

### P4：数据采集与世界模型，1 到 2 周

- 采集成功、失败、边界和反事实动作。
- 训练对象中心世界模型 ensemble。
- 验证一步预测、多步 rollout 和概率校准。
- 接入技能图搜索和 CEM。

### P5：VLM Proposal，1 周

- 输入场景摘要、对象图、能力和目标。
- 输出少量 JSON 技能骨架。
- 对模型输出执行严格 schema 校验。
- VLM 失败时回退到模板候选生成器。

### P6：闭环 Demo，1 周

- 每个技能结束后重新感知和规划。
- 实现对象 ID 关联和执行偏差检测。
- 展示候选计划、世界模型评分和最终选择。
- 在未见尺寸、质量和位置上测试。

### P7：CLIMB 与高台 Demo，2 到 3 周

- 训练攀爬技能。
- 加入支撑稳定性预测。
- 完成“推箱到高台前再攀爬”。
- 验证 Capability 变化导致规划方案变化。

完整 MVP 预计 6 到 8 周；加入可靠 CLIMB 后预计 8 到 11 周。主要进度风险来自 PUSH 和 CLIMB 的技能成功率，而不是世界模型网络本身。

## 12. 验收指标

MVP 至少报告：

- Held-out 场景任务成功率。
- NAV 和 PUSH 单技能成功率。
- 环境重构后的路径可达率。
- 非法碰撞率。
- 平均重规划次数。
- 世界模型一步状态预测误差。
- Skill Success 的 AUROC、Brier Score 和 ECE。
- 世界模型计划与 MuJoCo Oracle 计划的一致率。
- 不同 Capability 下方案变化的正确率。
- 相比“只导航”和“规则式重构”的任务成功率提升。

建议 MVP 验收线：

```text
NAV 成功率 >= 95%
PUSH 技能成功率 >= 85%
Blocked Passage 任务成功率 >= 80%
非法碰撞率 <= 10%
不可达场景正确拒绝率 >= 85%
```

## 13. Baseline 与消融实验

### 13.1 Baseline

1. 仅导航，不允许环境重构。
2. 手写规则：找到挡路箱体并推到固定位置。
3. VLM 直接生成计划，不使用世界模型验证。
4. 世界模型规划，不使用 VLM。
5. VLM + 世界模型 + 闭环重规划。
6. MuJoCo Oracle rollout 性能上界。

### 13.2 消融实验

- 去掉 Capability。
- 去掉 Object Tokens。
- 去掉 BEV。
- 去掉 ensemble uncertainty。
- 开环执行完整技能序列与逐技能重规划对比。
- 真值几何与 RGB-D 重建对比。
- 固定技能参数与 CEM 优化参数对比。

## 14. 推荐代码目录

```text
mujoco/reconfigurable_navigation/
  configs/
    scenarios.yaml
    capabilities.yaml
    world_model.yaml
  envs/
    reconfigurable_env.py
    scene_generator.py
    labels.py
  perception/
    rgbd_renderer.py
    bev_builder.py
    object_tracker.py
  representations/
    observation.py
    action.py
    object_graph.py
  skills/
    base.py
    navigate.py
    push.py
    climb.py
    jump.py
    stop.py
  data/
    collect.py
    dataset.py
    replay_buffer.py
  world_model/
    encoders.py
    dynamics.py
    heads.py
    ensemble.py
    train.py
  planning/
    proposal.py
    grounding.py
    skill_graph.py
    cem.py
    scorer.py
  vlm/
    client.py
    prompts.py
    schemas.py
  runtime/
    executor.py
    replanner.py
    safety.py
  visualization/
    dashboard.py
  evaluation/
    benchmark.py
    metrics.py
    baselines.py
```

## 15. 风险与约束

### 15.1 技能可靠性

如果底层技能失败率过高，世界模型只能学习不稳定动力学，高层规划结果也无法可靠执行。应先建立技能级测试集和明确终止条件。

### 15.2 数据偏差

只采集规则规划器选择的动作会造成严重选择偏差。需要主动采集失败候选、能力边界和模型高不确定度动作。

### 15.3 多步误差累积

世界模型 rollout 越长，预测误差越大。第一版将规划深度限制为 3 到 5 个技能，并使用逐技能重规划。

### 15.4 感知与动力学耦合

如果一开始直接使用 noisy RGB-D，难以判断失败来自感知、规划还是技能。开发顺序必须从 Oracle 状态逐步替换为视觉状态。

### 15.5 VLM 不确定性

VLM 只负责提出候选，不负责安全判断。所有候选必须经过 schema、几何、Capability 和世界模型验证。

### 15.6 Sim-to-Real

真实部署前需要加入质量、摩擦、接触刚度、相机和状态估计随机化，并设置保守的碰撞、推力和失稳保护。

## 16. 推荐技术主线

项目应按照以下顺序推进：

```text
可靠技能库
-> Oracle 高层闭环
-> 结构化技能级数据
-> 对象中心世界模型
-> 技能搜索与 CEM
-> VLM 候选提案
-> RGB-D 感知替换特权状态
-> Sim-to-Real
```

第一版应锁定 Blocked Passage。只要系统能稳定展示“原本不可达、识别可移动箱体、预测推箱结果、执行环境重构、重新导航到目标”，就构成一个完整且有说服力的世界模型可重构导航 Demo。
