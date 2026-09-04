# GO2-ARX5 MuJoCo 训练

该目录将项目的 GO2-ARX5 Flat 任务迁移为 Gymnasium 环境，使用 MuJoCo 物理仿真和 Stable-Baselines3 PPO 训练，不依赖 Isaac Sim 或 Isaac Lab。

## 安装

在已经安装项目 MuJoCo 依赖的环境中执行：

```bash
python -m pip install -r mujoco/train/go2_arx5/requirements.txt
```

## 检查环境

```bash
python mujoco/train/go2_arx5/check_env.py
```

## 短训练测试

先用单环境确认训练链路：

```bash
python mujoco/train/go2_arx5/train.py --num-envs 1 --total-timesteps 10000
```

## 可视化

训练时实时显示机器人（只支持单环境，会明显降低训练速度）：

```bash
python mujoco/train/go2_arx5/train.py \
  --num-envs 1 \
  --total-timesteps 10000 \
  --render
```

正式训练建议不加 `--render`，同时在另一个终端查看 TensorBoard：

```bash
tensorboard --logdir runs/mujoco/go2_arx5/tensorboard
```

浏览器打开 `http://localhost:6006`，可以查看 PPO 损失、总奖励以及各奖励分量。

## 正式训练

```bash
python mujoco/train/go2_arx5/train.py --num-envs 4 --total-timesteps 5000000
```

CPU 核数充足时可提高 `--num-envs`。训练默认使用小范围速度和末端位置命令；基础策略收敛后，可以用 `--command-profile full` 扩大命令范围继续训练。

继续训练：

```bash
python mujoco/train/go2_arx5/train.py \
  --resume runs/mujoco/go2_arx5/final_model.zip \
  --command-profile full \
  --num-envs 4 \
  --total-timesteps 5000000
```

## 评估结果

```bash
tensorboard --logdir runs/mujoco/go2_arx5/tensorboard
python mujoco/train/go2_arx5/play.py runs/mujoco/go2_arx5/best/best_model.zip
```

训练输出包括：

- `checkpoints/`：定期保存的完整 PPO checkpoint。
- `best/best_model.zip`：评估奖励最高的完整 checkpoint。
- `final_model.zip`：训练结束时的完整 checkpoint。
- `exported/policy.pt`：210 维观测到 18 维动作的 TorchScript 推理模型。
- `tensorboard/`：训练曲线。

`policy.pt` 只用于推理；继续训练应使用 `.zip` checkpoint。
