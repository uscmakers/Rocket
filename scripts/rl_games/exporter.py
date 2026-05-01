# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX export for rl_games policies.

isaaclab_rl's built-in exporter (isaaclab_rl.rsl_rl.exporter) is RSL-RL specific
and incompatible with rl_games — it expects policy.is_recurrent and policy.actor
which rl_games Network objects do not have.  This module exports directly via
torch.onnx using the same opset and dummy-input pattern as the isaaclab exporter.
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import torch
import torch.nn as nn
import yaml

from rl_games.common import env_configurations, vecenv
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


class _ActorWrapper(nn.Module):
    """Wraps an rl_games model for deterministic inference: obs -> action means.

    rl_games models expect a dict input and return a dict.  This wrapper
    presents the flat tensor interface that torch.onnx.export requires.
    Input normalisation (running_mean_std) is embedded inside the rl_games
    model and is therefore included in the exported graph automatically.
    """

    def __init__(self, rlg_model: nn.Module):
        super().__init__()
        self.model = rlg_model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        result = self.model({"obs": obs, "is_train": False})
        return result["mus"]


def export_trained_policy_to_onnx(log_root_path: str, log_dir: str, config_name: str, agent_cfg: dict) -> None:
    """Export the best saved rl_games checkpoint to ONNX.

    Loads config_name.pth (the best checkpoint written by rl_games during
    training) via create_player + restore, then exports to ONNX using
    torch.onnx.export with opset 18.

    Args:
        log_root_path: Root logging directory (e.g. logs/rl_games/<name>).
        log_dir:       Run subdirectory (e.g. 2024-01-01_12-00-00).
        config_name:   Agent config name — also the checkpoint stem (e.g. rocket_direct).
        agent_cfg:     Full agent config dict passed to Runner.load().
    """
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    export_dir = os.path.join(log_root_path, log_dir, "nn")
    os.makedirs(export_dir, exist_ok=True)

    best_ckpt = os.path.join(export_dir, f"{config_name}.pth")
    if not os.path.exists(best_ckpt):
        print(f"[WARNING] Best checkpoint not found at {best_ckpt}, skipping ONNX export.")
        return

    # Load best checkpoint using the player interface (same as play.py).
    export_runner = Runner(IsaacAlgoObserver())
    export_runner.load(agent_cfg)
    export_agent = export_runner.create_player()
    export_agent.restore(best_ckpt)
    export_agent.reset()

    obs_size = int(export_agent.obs_shape[0])
    wrapper = _ActorWrapper(export_agent.model).cpu().eval()
    dummy_obs = torch.zeros(1, obs_size)

    onnx_path = os.path.join(export_dir, f"{config_name}.onnx")
    torch.onnx.export(
        wrapper,
        dummy_obs,
        onnx_path,
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] Exported ONNX to: {onnx_path}")


def _load_agent_cfg(agent_cfg_path: str) -> dict:
    with open(agent_cfg_path, "r", encoding="utf-8") as handle:
        agent_cfg = yaml.safe_load(handle)
    return agent_cfg or {}


def _register_rlgpu_env(env, rl_device: str, agent_cfg: dict) -> None:
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an rl_games .pth checkpoint to ONNX.")
    parser.add_argument("--policy", required=True, help="Path to the .pth checkpoint.")
    parser.add_argument(
        "--agent-cfg",
        default=str(
            Path(__file__).resolve().parents[2]
            / "source"
            / "rocket"
            / "rocket"
            / "tasks"
            / "direct"
            / "rocket"
            / "agents"
            / "rl_games_ppo_cfg.yaml"
        ),
        help="Path to the rl_games agent YAML.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .onnx path. Defaults to rocket_direct.onnx next to the policy.",
    )
    parser.add_argument("--task", type=str, default="Template-Rocket-Direct-v0", help="Name of the task.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rl_games_cfg_entry_point",
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(args_cli: argparse.Namespace) -> None:
    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import rocket.tasks  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def _run_export(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict):
        policy_path = os.path.abspath(args_cli.policy)
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Checkpoint not found: {policy_path}")

        agent_cfg = _load_agent_cfg(args_cli.agent_cfg)
        agent_cfg.setdefault("params", {})
        agent_cfg["params"].setdefault("env", {})
        agent_cfg["params"].setdefault("config", {})

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # remove camera configs if enable cameras is not enabled
        if not args_cli.enable_cameras and hasattr(env_cfg.scene, "tiled_camera"):
            del env_cfg.scene.tiled_camera

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        rl_device = agent_cfg["params"]["config"].get("device", env_cfg.sim.device)
        _register_rlgpu_env(env, rl_device, agent_cfg)

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = policy_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        runner = Runner()
        runner.load(agent_cfg)
        export_agent: BasePlayer = runner.create_player()
        export_agent.restore(policy_path)
        export_agent.reset()

        obs_size = int(export_agent.obs_shape[0])
        wrapper = _ActorWrapper(export_agent.model).cpu().eval()
        dummy_obs = torch.zeros(1, obs_size)

        policy_dir = os.path.dirname(policy_path)
        output_path = args_cli.output or os.path.join(policy_dir, "rocket_direct.onnx")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        torch.onnx.export(
            wrapper,
            dummy_obs,
            output_path,
            export_params=True,
            opset_version=18,
            verbose=False,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
        print(f"[INFO] Exported ONNX to: {output_path}")

        env.close()

    _run_export()
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Checkpoint not found: {policy_path}")

    agent_cfg = _load_agent_cfg(args_cli.agent_cfg)
    agent_cfg.setdefault("params", {})
    agent_cfg["params"].setdefault("env", {})
    agent_cfg["params"].setdefault("config", {})

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # remove camera configs if enable cameras is not enabled
    if not args_cli.enable_cameras and hasattr(env_cfg.scene, "tiled_camera"):
        del env_cfg.scene.tiled_camera

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    rl_device = agent_cfg["params"]["config"].get("device", env_cfg.sim.device)
    _register_rlgpu_env(env, rl_device, agent_cfg)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = policy_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    export_agent: BasePlayer = runner.create_player()
    export_agent.restore(policy_path)
    export_agent.reset()

    obs_size = int(export_agent.obs_shape[0])
    wrapper = _ActorWrapper(export_agent.model).cpu().eval()
    dummy_obs = torch.zeros(1, obs_size)

    policy_dir = os.path.dirname(policy_path)
    output_path = args_cli.output or os.path.join(policy_dir, "rocket_direct.onnx")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] Exported ONNX to: {output_path}")

    env.close()


if __name__ == "__main__":
    parser = _build_parser()
    args_cli, hydra_args = parser.parse_known_args()

    # clear out sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    main(args_cli)
    simulation_app.close()
