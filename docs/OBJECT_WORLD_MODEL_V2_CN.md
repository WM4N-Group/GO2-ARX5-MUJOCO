# 对象中心世界模型 v2

日期：2026-09-15。先将多布局、后续标签和MLP基线发布为a748e53，再完成本页对象模型工作。最新发布以Git历史为准，数据和训练版本以文件hash为准。原29/32技能基线及三份actor保持冻结，当前仍由规则Oracle执行。

## 1. 输入与网络

[spatial_encoding.py](../mujoco/reconfigurable_navigation/world_model/spatial_encoding.py) 定义独立schema `box_support_objects_bev_v2`：

- BEV为8x128x128、5cm网格，中心为机器人XY，列方向+X、行方向+Y，保持世界轴对齐。
- 通道为占据、米制顶面高度、高度边缘代理、几何可支撑、可移动、静态障碍、已知区域及机器人/目标提示。
- BEV由特权状态中的yaw旋转长方体构建，假设当前场景的无限平地完整已知；不是RGB-D重建，也不是完整roll/pitch网格地形。几何可支撑通道不等于已验证四足稳定。
- 对象最多32个，每个32维：保留原中心、尺寸、语义和置信度，并加入yaw、线/角速度、质量、摩擦及已知位、接触和对象/支撑指针。有效mask和对象ID独立保存。
- 本体补充为18个关节位置、18个速度、18维上一动作、18维当前PD目标，以及一位有效标记。都是技能开始前的信息，不读取终态、结果或数据集分组。

[ObjectWorldModel](../mujoco/reconfigurable_navigation/world_model/object_model.py) 使用四层CNN（16/32/64/128通道）、一层对象自注意力和四层Transformer dynamics，宽128、4 heads、FFN宽256、dropout0。Scene、Robot、Goal、Capability和Action分别形成token，保留对象级表示，不提前整体平均。对象置换和填充不会改变预测语义。

输出仍沿用可直接对照的12个回归量和3个概率头：机器人/箱体位置及yaw增量、技能和总尝试耗时、技能成功、非法碰撞、指定规则后续任务成功。没有新增无可靠负标签的全局不可达分类，也尚未覆盖原方案全部支撑/终态预测头。

统计先验只由训练组按技能与目标类型拟合，网络预测残差，输出头从零开始。同信息残差MLP使用两层128单元、相同本体信息和先验作消融。不能将所有增益单独归因于Transformer，因为网络容量和结构也不同。

## 2. 扩样与数据身份

配置 [object_world_model_v2.json](../mujoco/experiments/object_world_model_v2.json) 固定12个布局。每个布局5个起点，按配置顺序使用互不重叠的700-759种子；对齐/左偏/右偏族分别为train/validation/test。

[collect_object_world_data.py](../mujoco/collect_object_world_data.py) 在每个已接受技能起点创建独立内存快照，评估候选及规则后续，再继续现场原动作。每次检查现场integration state不变，并逐字段核对reference转移与真实原动作；已终止reference的整任务结果和总耗时也与现场回合核对。

60个教师回合43成功，产生250个技能边界、1500请求：1130技能成功、120物理失败、250拒绝。1250条实际执行样本分为train490/validation420/test340，PUSH300、NAV475、CLIMB475；规则后续1028成功、222失败。该扩展分布结果不替代原固定布局29/32验收。

每个训练边界有稳定 `snapshot_id`，由场景/seed/技能序号、物理代码和integration state绑定；它不是归档文件SHA。只保存每个族首个回合的代表归档，共11个，其余239个边界明确标记无归档。无归档记录需按场景、seed及版本重建，不能直接使用文件快照重放。

模型加载器校验来源文件hash、before/action一致性和冻结teacher契约，并按场景、回合及边界身份阻止跨集合。拒绝和未知标签不伪造动力学监督。新候选ID还包含动作和预算，避免不同预算的记录混为同一实验。

## 3. 固定对照结果

固定seed7，AdamW学习率3e-4、权重衰减1e-3、batch32、最多200轮、validation耐心30轮。零残差统计先验作为第0轮候选，只有validation改善才选择学习残差；测试集只用于最终报告。残差MLP选第6轮，对象Transformer选第18轮。

| 测试指标 | 统计基线 | 残差MLP | 对象Transformer |
| --- | --- | --- | --- |
| 机器人位置RMSE | 0.160m | 0.164m | 0.164m |
| 箱体位置RMSE | 0.0642m | 0.0627m | 0.0601m |
| 非法碰撞Brier | 0.0919 | 0.0925 | 0.0597 |
| 规则任务成功Brier | 0.1516 | 0.1533 | 0.1319 |
| 总尝试耗时MAE | 4.255s | 4.156s | 3.441s |
| 选中候选的实际任务成功数 | 55/68 | 48/68 | 56/68 |

当前候选集最佳可观察结果是65/68，不是穷尽可达性上界。选择分数固定为任务概率减碰撞概率，再减0.001倍尝试耗时；统计基线同类候选同分时取reference。

这是单次初始化、同一有限场景族上的离线结果。对象模型的概率和成本指标较好，选择比统计基线仅多成功1次，机器人位置误差仍未优于统计基线，不能宣称显著增益或接管规划。v1的48测试样本与本页340样本不同，不能直接比较两个百分比作为架构收益。

## 4. 验证与数值边界

22项旧/新模型契约检查通过，包括BEV几何、对象置换、填充、起点隔离、训练组归一化、统计先验、本体有效性和归档身份。两个网络的CUDA前后向、checkpoint保存恢复通过；六个代表物理归档在MuJoCo中零容差重放通过。

Transformer未通过最初设定的CPU/GPU严格张量容差（rtol2e-5/atol2e-6）。相同张量对照确认差异来自设备运算内核，GPU重载复现原预测。全部1250条CPU/GPU预测的最大实际差异：位置3.34e-5m、yaw2.67e-5rad、时间0.00188s、概率0.000145；250组候选的选择一致。该验证不是位级一致，详细结果保留在 `numeric-validation.json`。

## 5. 产物与复现

本地 `/home/yuanyue/re-nav/artifacts/object-world-model-v2-20260915/`，服务器 `/mnt/yuanyue/data/object-world-model-v2-20260915/`。主要入口是 `data/manifest.json`、`comparison-seed7/comparison.json`、两个模型子目录内的 `model.pt/report.json/predictions.jsonl` 和数值验证报告。JSON和模型已在本地，代表性物理归档保留服务器，训练产物不纳入Git。

源码部署前备份到 `/mnt/yuanyue/backups/object-world-v2-iusmko/`，未pull/reset服务器旧工作树。训练使用GPU0的现有CUDA PyTorch环境，没有启动Isaac Sim或替换MuJoCo环境；GPU1上的已有任务未改动。

在服务器仓库根目录运行，输出目录必须不存在：

```bash
export DATA_ROOT=/mnt/yuanyue/data/object-world-reproduction
export BOX_BUNDLES=/mnt/yuanyue/data/box-skills-eval
env ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/yuanyue/envs/go2-mujoco/bin/python mujoco/collect_object_world_data.py \
  --push-policy "$BOX_BUNDLES/push-height020-stop200-bundle/policy.pt" \
  --climb-policy "$BOX_BUNDLES/climb-prepared-ground499-bundle/policy.pt" \
  --platform-policy "$BOX_BUNDLES/climb-prepared-gaps399-bundle/policy.pt" \
  --suite-json mujoco/experiments/object_world_model_v2.json \
  --output-dir "$DATA_ROOT/data" --seed-offset 700 --seeds 5 \
  --candidates-per-snapshot 6 --candidate-seed 91 --workers 8 --archive-first-per-family

env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /mnt/yuanyue/envs/go2-isaac/bin/python mujoco/train_object_world_model.py \
  --manifest "$DATA_ROOT/data/manifest.json" --output-dir "$DATA_ROOT/comparison-seed7" \
  --device cuda:0 --seed 7 --width 128 --layers 4 --heads 4 \
  --epochs 200 --patience 30 --batch-size 32 --learning-rate 0.0003
```

下一项是重复初始化与ensemble、风险校准和多步误差检查，随后才考虑shadow旁路。当前没有ensemble、学习模型闭环、CEM在线接管、RGB-D或VLM推理，低层技能不重训。