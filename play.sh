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

# Usage: ./play.sh --standing | --walking [--roboland]
source policies.cfg

# Parse arguments
POLICY=""
ROBOLAND=false
for arg in "$@"; do
    case "$arg" in
        --standing) POLICY="standing" ;;
        --walking)  POLICY="walking" ;;
        --roboland) ROBOLAND=true ;;
        *) echo "Unknown argument: $arg"; echo "Usage: $0 --standing | --walking [--roboland]"; exit 1 ;;
    esac
done

if [ -z "$POLICY" ]; then
    echo "Usage: $0 --standing | --walking [--roboland]"
    exit 1
fi

# Select run dir and checkpoint epoch based on policy
if [ "$POLICY" = "standing" ]; then
    RUN_DIR="$STANDING_POLICY_CHECKPOINT"
    CHECKPOINT_EPOCH="$STANDING_POLICY_CHECKPOINT_EPOCH"
else
    RUN_DIR="$WALKING_POLICY_CHECKPOINT"
    CHECKPOINT_EPOCH="$WALKING_POLICY_CHECKPOINT_EPOCH"
fi

# Resolve checkpoint path
LOG_DIR="logs/rl_games/rocket_direct/$RUN_DIR/nn"
CHECKPOINT="$LOG_DIR/rocket_direct.pth"

if [ -n "$CHECKPOINT_EPOCH" ] && [ "$CHECKPOINT_EPOCH" != "none" ]; then
    # Find the file matching last_rocket_direct_ep_{epoch}_*.pth
    CHECKPOINT=$(ls "$LOG_DIR"/last_rocket_direct_ep_${CHECKPOINT_EPOCH}_*.pth 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found for epoch $CHECKPOINT_EPOCH in $LOG_DIR"
        exit 1
    fi
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

echo "Policy:     --$POLICY"
echo "Checkpoint: $CHECKPOINT"

# Source user credentials
if [ -f ~/setup.sh ]; then
    source ~/setup.sh
else
    echo "WARNING: ~/setup.sh not found! Credentials will not be loaded"
fi

if [ "$ROBOLAND" = true ]; then
    # Run locally with custom conda/venv environment
    source ~/env_isaaclab/bin/activate
    python3 -u scripts/rl_games/play.py \
        --checkpoint "$CHECKPOINT" \
        --num_envs 1
else
    # Run on HPC via Apptainer
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
fi

echo "Job completed"
