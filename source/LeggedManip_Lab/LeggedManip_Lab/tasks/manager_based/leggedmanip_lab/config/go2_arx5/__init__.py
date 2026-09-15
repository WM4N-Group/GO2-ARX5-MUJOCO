import gymnasium as gym
from . import agents

# Flat
gym.register(
    id="GO2-ARX5-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2ARX5FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5FlatPPORunnerCfg",
    },
)

gym.register(
    id="GO2-ARX5-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2ARX5FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5FlatPPORunnerCfg",
    },
)

# Climb
gym.register(
    id="GO2-ARX5-Climb",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.climb_env_cfg:Go2ARX5ClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5ClimbPPORunnerCfg",
    },
)

gym.register(
    id="GO2-ARX5-Climb-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.climb_env_cfg:Go2ARX5ClimbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5ClimbPPORunnerCfg",
    },
)

for task_id, config_name in (
    ("GO2-ARX5-Box-Climb", "Go2ARX5BoxClimbEnvCfg"),
    ("GO2-ARX5-Box-Climb-Play", "Go2ARX5BoxClimbEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.box_climb_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5BoxClimbPPORunnerCfg",
        },
    )

for task_id, config_name in (
    ("GO2-ARX5-Box-Push", "Go2ARX5BoxPushEnvCfg"),
    ("GO2-ARX5-Box-Push-Play", "Go2ARX5BoxPushEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.box_push_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5BoxPushPPORunnerCfg",
        },
    )

for task_id, config_name in (
    ("GO2-ARX5-Box-Push-Hybrid", "Go2ARX5BoxPushHybridEnvCfg"),
    ("GO2-ARX5-Box-Push-Hybrid-Play", "Go2ARX5BoxPushHybridEnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.box_push_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5BoxPushHybridPPORunnerCfg",
        },
    )

# WBC
gym.register(
    id="GO2-ARX5-WBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:Go2ARX5WBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5WBCPPORunnerCfg",
    },
)

gym.register(
    id="GO2-ARX5-WBC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:Go2ARX5WBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ARX5WBCPPORunnerCfg",
    },
)

