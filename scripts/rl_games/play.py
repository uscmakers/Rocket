# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import functools
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Rocket-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log play diagnostics to Weights & Biases.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import random
import time
import torch
import numpy as np
import functools
torch.load = functools.partial(torch.load, weights_only=False)

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rocket.tasks  # noqa: F401


import math as _math

# Joint order matches _joint_ids = servo_joint_ids + stepper_joint_ids
# servo_joint_names  = ["Revolute1", "Revolute2"]  → hip_servo_L, hip_servo_R
# hip_joint_names    = ["Revolute3", "Revolute4"]  → hip_stepper_L, hip_stepper_R
# knee_joint_names   = ["Revolute5", "Revolute6"]  → knee_stepper_L, knee_stepper_R
_JOINT_LABELS = [
    "hip_servo_L", "hip_servo_R",
    "hip_stepper_L", "hip_stepper_R",
    "knee_stepper_L", "knee_stepper_R",
]


class PlayLogger:
    """Logs joint state (pos, vel, torque) and foot diagnostics for env 0 to wandb."""

    def __init__(self, wandb, step_ref: list):
        self._wandb = wandb
        self._step  = step_ref  # shared [int] so all loggers stay in sync

    def log(self, env):
        rocket_env = env.unwrapped
        joint_ids  = rocket_env._joint_ids

        joint_pos = rocket_env.robot.data.joint_pos[0, joint_ids].cpu()
        joint_vel = rocket_env.robot.data.joint_vel[0, joint_ids].cpu()
        torques   = rocket_env.robot.data.applied_torque[0, joint_ids].cpu()

        data = {}
        for i, label in enumerate(_JOINT_LABELS):
            data[f"joint_pos/{label}"]    = joint_pos[i].item()
            data[f"joint_vel/{label}"]    = joint_vel[i].item()
            data[f"joint_torque/{label}"] = torques[i].item()

        data["joint_vel/max_abs"]    = joint_vel.abs().max().item()
        data["joint_torque/max_abs"] = torques.abs().max().item()

        # foot clearance and air/contact time (L=index 0, R=index 1)
        toe_pos_z    = rocket_env.robot.data.body_pos_w[0, rocket_env._toe_body_ids, 2].cpu()
        air_time     = rocket_env.contact_sensor_toes.data.current_air_time[0].cpu()
        contact_time = rocket_env.contact_sensor_toes.data.current_contact_time[0].cpu()

        for i, side in enumerate(["L", "R"]):
            data[f"foot/toe_clearance_m/{side}"] = toe_pos_z[i].item()
            data[f"foot/air_time_s/{side}"]       = air_time[i].item()
            data[f"foot/contact_time_s/{side}"]   = contact_time[i].item()

        self._wandb.log(data, step=self._step[0])


_TRACKING_FLUSH_EVERY = 50  # steps between chart flushes


class DiagnosticsLogger:
    """Logs robot-level diagnostics and per-joint position tracking (actual vs target).

    Tracking charts: uses wandb.plot.line_series so actual_rad and target_rad are
    automatically overlaid on the same graph — no manual panel config needed.
    Charts flush every _TRACKING_FLUSH_EVERY steps.
    """

    def __init__(self, wandb, step_ref: list):
        self._wandb = wandb
        self._step  = step_ref
        # rolling buffers for line_series charts
        self._buf_steps:  list = []
        self._buf_actual: dict = {label: [] for label in _JOINT_LABELS}
        self._buf_target: dict = {label: [] for label in _JOINT_LABELS}

    def log(self, env):
        rocket_env = env.unwrapped
        data = {}

        # --- robot state ---
        quat = rocket_env.imu.data.quat_w[0].cpu()  # (w, x, y, z)
        qw, qx, qy, qz = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()

        # forward tilt: angle of body Z in world Y-Z plane (positive = leaning toward +Y)
        body_z_y = 2.0 * (qy * qz - qw * qx)
        data["robot/forward_tilt_deg"] = _math.degrees(_math.asin(max(-1.0, min(1.0, body_z_y))))

        data["robot/height_m"] = rocket_env.robot.data.root_pos_w[0, 2].item()

        # 1.0 when both feet off ground, 0.0 otherwise
        contact_time = rocket_env.contact_sensor_toes.data.current_contact_time[0].cpu()
        data["robot/both_feet_airborne"] = 1.0 if (contact_time[0] == 0.0 and contact_time[1] == 0.0) else 0.0

        self._wandb.log(data, step=self._step[0])

        # --- buffer per-joint tracking data ---
        joint_ids  = rocket_env._joint_ids
        actual_pos = rocket_env.robot.data.joint_pos[0, joint_ids].cpu()
        target_pos = rocket_env._delta_target_pos[0].cpu()

        self._buf_steps.append(self._step[0])
        for i, label in enumerate(_JOINT_LABELS):
            self._buf_actual[label].append(actual_pos[i].item())
            self._buf_target[label].append(target_pos[i].item())

        # flush as line_series charts every N steps so actual+target appear on same graph
        if len(self._buf_steps) >= _TRACKING_FLUSH_EVERY:
            charts = {}
            for label in _JOINT_LABELS:
                charts[f"tracking/{label}"] = self._wandb.plot.line_series(
                    xs=self._buf_steps,
                    ys=[self._buf_actual[label], self._buf_target[label]],
                    keys=["actual_rad", "target_rad"],
                    title=label,
                    xname="step",
                )
            self._wandb.log(charts, step=self._step[0])
            # clear buffers
            self._buf_steps.clear()
            for label in _JOINT_LABELS:
                self._buf_actual[label].clear()
                self._buf_target[label].clear()


class WandbSession:
    """Owns the wandb run and coordinates all sub-loggers."""

    def __init__(self, run_name: str):
        import wandb
        wandb.init(project="rocket-play", name=run_name, reinit=True)
        self._wandb   = wandb
        self._step    = [0]  # shared mutable int across loggers
        self.play     = PlayLogger(wandb, self._step)
        self.diag     = DiagnosticsLogger(wandb, self._step)

    def log(self, env):
        self.play.log(env)
        self.diag.log(env)
        self._step[0] += 1

    def finish(self):
        self._wandb.finish()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # remove camera configs if enable cameras is not enabled
    if not args_cli.enable_cameras and hasattr(env_cfg.scene, "tiled_camera"):
        del env_cfg.scene.tiled_camera

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # set up wandb logger if requested
    play_logger = WandbSession(run_name=log_dir) if args_cli.wandb else None

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0

    # delta tracking state
    _track_intended = None
    _track_prev_pos = None
    _track_ratio_acc = torch.zeros(6)
    _track_error_acc = torch.zeros(6)
    _track_n = 0
    JOINT_LABELS = ["srv_L", "srv_R", "hip_L", "hip_R", "kne_L", "kne_R"]
    PRINT_EVERY = 50
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs)

            # capture intended target and current pos before step
            _re = env.unwrapped
            _track_prev_pos    = _re.robot.data.joint_pos[0, _re._joint_ids].clone().cpu()
            _track_intended    = _re._delta_target_pos[0].clone().cpu()

            # env stepping
            obs, _, dones, _ = env.step(actions)

            # compute tracking stats
            actual_pos     = _re.robot.data.joint_pos[0, _re._joint_ids].clone().cpu()
            intended_delta = _track_intended - _track_prev_pos
            actual_delta   = actual_pos - _track_prev_pos
            mask = intended_delta.abs() > 1e-3
            ratio = torch.zeros(6)
            ratio[mask] = (actual_delta[mask] / intended_delta[mask]).clamp(-2, 2)
            _track_ratio_acc += ratio
            _track_error_acc += (actual_pos - _track_intended).abs()
            _track_n += 1
            if _track_n % PRINT_EVERY == 0:
                r = _track_ratio_acc / PRINT_EVERY
                e = _track_error_acc / PRINT_EVERY
                print(f"\n--- delta tracking (avg over {PRINT_EVERY} steps) ---")
                print(f"  {'joint':<8} {'achieved%':>10} {'err_rad':>9} {'status':>10}")
                for i, lbl in enumerate(JOINT_LABELS):
                    pct = r[i].item() * 100
                    status = "OK" if 80 <= pct <= 120 else ("OVERSHOOT" if pct > 120 else "UNDERSHOOT")
                    print(f"  {lbl:<8} {pct:>9.1f}% {e[i].item():>9.4f}  {status}")
                _track_ratio_acc.zero_()
                _track_error_acc.zero_()
            print(f"obs: {obs}\nactions: {actions}")

            if play_logger is not None:
                play_logger.log(env)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if play_logger is not None:
        play_logger.finish()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
