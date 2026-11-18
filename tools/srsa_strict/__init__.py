# [E3] Registers the strict sparse-reward env (a modification of the paper's code). Original SRSA / Isaac Lab files are not changed.
# Imported by train_strict.py (a copy of SRSA scripts/rl_games/train.py plus one import line for this package).
import gymnasium as gym
gym.register(
    id="Assembly-StrictSparse-Sil-v0",
    entry_point="srsa_strict.assembly_strict_env:AssemblyStrictSparseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.automate.assembly_env_cfg:AssemblyEnvCfg",
        "rl_games_cfg_entry_point": "SRSA.tasks.direct.srsa.agents:rl_games_ppo_sil_cfg.yaml",
    },
)
