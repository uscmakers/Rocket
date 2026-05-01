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

import os
import torch
import torch.nn as nn


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
