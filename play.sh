#!/bin/bash
#SBATCH --job-name=rocket-play
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"  # Exclude P100
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/rocket-play_%j.out
#SBATCH --error=jobs/rocket-play_%j.err

# Load any required modules (adjust based on your HPC)
# module load apptainer  # Uncomment if needed

# Source user credentials from setup.sh
if [ -f ~/setup.sh ]; then
    source ~/setup.sh
else
    echo "ERROR: ~/setup.sh not found!"
    echo "Create ~/setup.sh with your credentials to log runs to wandb"
fi

# Set environment variables
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
# Path to your SIF file (adjust this path!)
SIF_PATH="$HOME/isaac-sim_5.1.0.sif"
CHECKPOINT_NAME="2026-02-19_18-11-41"
CHECKPOINT_PATH="$HOME/Rocket/logs/rl_games/rocket_direct/$CHECKPOINT_NAME/nn/rocket_direct.pth"

# Run Isaac Sim with Apptainer
module load apptainer
apptainer --version

# Create cache directories (ESSENTIAL)
mkdir -p ~/isaac-sim-cache/data
mkdir -p ~/isaac-sim-cache/cache
mkdir -p ~/isaac-sim-cache/logs

# Run your script with necessary binds
apptainer exec --nv \
    --bind ~/isaac-sim-cache/data:/isaac-sim/kit/data \
    --bind ~/isaac-sim-cache/cache:/isaac-sim/kit/cache \
    --bind ~/isaac-sim-cache/logs:/isaac-sim/kit/logs \
    $SIF_PATH \
    bash -c "
    /isaac-sim/python.sh -m pip install --user --no-build-isolation "isaaclab[all]==2.3.2" --extra-index-url https://pypi.nvidia.com &&
    /isaac-sim/python.sh -m pip install --user --force-reinstall -e source/rocket &&
    /isaac-sim/python.sh -m pip install --user rl-games &&
    /isaac-sim/python.sh -m pip install --user imageio imageio-ffmpeg &&
    /isaac-sim/python.sh -u scripts/list_envs.py &&
    /isaac-sim/python.sh -u scripts/rl_games/play.py --headless --video --checkpoint "$CHECKPOINT_PATH"
    "

echo "Job completed"