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
from pathlib import Path

import torch
import torch.nn as nn
import yaml


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


def export_checkpoint_to_onnx(policy_path: str, agent_cfg_path: str, output_path: str) -> None:
    """Export a specific rl_games checkpoint to ONNX.

    Args:
        policy_path:     Path to the .pth checkpoint.
        agent_cfg_path:  Path to the rl_games agent YAML.
        output_path:     Full path to the output .onnx file.
    """
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    policy_path = os.path.abspath(policy_path)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Checkpoint not found: {policy_path}")

    with open(agent_cfg_path, "r", encoding="utf-8") as handle:
        agent_cfg = yaml.safe_load(handle)

    agent_cfg = agent_cfg or {}
    agent_cfg.setdefault("params", {})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = policy_path

    export_runner = Runner(IsaacAlgoObserver())
    export_runner.load(agent_cfg)
    export_agent = export_runner.create_player()
    export_agent.restore(policy_path)
    export_agent.reset()

    obs_size = int(export_agent.obs_shape[0])
    wrapper = _ActorWrapper(export_agent.model).cpu().eval()
    dummy_obs = torch.zeros(1, obs_size)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an rl_games .pth checkpoint to ONNX.")
    parser.add_argument("--policy", required=True, help="Path to the .pth checkpoint.")
    parser.add_argument(
        "--agent-cfg",
        default=str(Path(__file__).resolve().parents[2] / "source" / "rocket" / "rocket" / "tasks" / "direct" / "rocket" / "agents" / "rl_games_ppo_cfg.yaml"),
        help="Path to the rl_games agent YAML.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .onnx path. Defaults to rocket_direct.onnx next to the policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy_path = os.path.abspath(args.policy)
    policy_dir = os.path.dirname(policy_path)
    output_path = args.output or os.path.join(policy_dir, "rocket_direct.onnx")
    export_checkpoint_to_onnx(policy_path, args.agent_cfg, output_path)


if __name__ == "__main__":
    main()
