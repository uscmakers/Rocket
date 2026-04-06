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

# Usage: ./play.sh --standing | --walking
source policies.cfg

# Parse policy type argument
case "$1" in
    --standing)
        RUN_DIR="$STANDING_POLICY_CHECKPOINT"
        CHECKPOINT_FILE="$STANDING_POLICY_CHECKPOINT_FILE"
        ;;
    --walking)
        RUN_DIR="$WALKING_POLICY_CHECKPOINT"
        CHECKPOINT_FILE="$WALKING_POLICY_CHECKPOINT_FILE"
        ;;
    *)
        echo "Usage: $0 --standing | --walking"
        exit 1
        ;;
esac

# Resolve checkpoint path
LOG_DIR="logs/rl_games/rocket_direct/$RUN_DIR/nn"
CHECKPOINT="$LOG_DIR/rocket_direct.pth"

if [ -n "$CHECKPOINT_FILE" ] && [ "$CHECKPOINT_FILE" != "none" ]; then
    CHECKPOINT="$LOG_DIR/$CHECKPOINT_FILE"
else
    # Auto-select the highest epoch checkpoint
    BEST_EPOCH=0
    for pth in "$LOG_DIR"/last_rocket_direct_ep_*.pth; do
        [ -f "$pth" ] || continue
        epoch=$(echo "$pth" | grep -oP '(?<=_ep_)\d+')
        if [ -n "$epoch" ] && [ "$epoch" -gt "$BEST_EPOCH" ]; then
            BEST_EPOCH=$epoch
            CHECKPOINT="$pth"
        fi
    done
fi

echo "Policy:     $1"
echo "Checkpoint: $CHECKPOINT"

# Source user credentials
if [ -f ~/setup.sh ]; then
    source ~/setup.sh
else
    echo "ERROR: ~/setup.sh not found!"
    exit 1
fi

# Environment
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
unset DISPLAY

SIF_PATH="$HOME/isaac-sim_5.1.0.sif"

module load apptainer

mkdir -p ~/isaac-sim-cache/data
mkdir -p ~/isaac-sim-cache/cache
mkdir -p ~/isaac-sim-cache/logs

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
