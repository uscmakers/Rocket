#!/bin/bash
#SBATCH --account=jessetho_1732
#SBATCH --job-name=rocket-play
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|l40s"  # Exclude P100
#SBATCH --mem=32G
#SBATCH --time=30:00
#SBATCH --output=jobs/rocket-play_%j.out
#SBATCH --error=jobs/rocket-play_%j.err

# Load any required modules (adjust based on your HPC)
# module load apptainer  # Uncomment if needed

# Check available accounts and fair share with sshare -l -U $USER

# Dynamically extract the last checkpoint for the current standing policy
source policies.cfg

LOG_DIR="logs/rl_games/rocket_direct/$STANDING_POLICY_CHECKPOINT/nn"

# default
CHECKPOINT="$LOG_DIR/rocket_direct.pth"
BEST_EPOCH=0

# scan for epoch-stamped checkpoints
for pth in "$LOG_DIR"/last_rocket_direct_ep_*.pth; do
    [ -f "$pth" ] || continue
    # extract epoch number from filename
    epoch=$(echo "$pth" | grep -oP '(?<=_ep_)\d+')
    if [ -n "$epoch" ] && [ "$epoch" -gt "$BEST_EPOCH" ]; then
        BEST_EPOCH=$epoch
        CHECKPOINT="$pth"
    fi
done

echo "Using checkpoint: $CHECKPOINT"

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
unset DISPLAY  # Ensure headless mode

# Path to your SIF file (adjust this path!)
SIF_PATH="$HOME/isaac-sim_5.1.0.sif"

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
    /isaac-sim/python.sh -u scripts/rl_games/play.py --num_envs 32 --headless --video --checkpoint "$CHECKPOINT"
    "

echo "Job completed"