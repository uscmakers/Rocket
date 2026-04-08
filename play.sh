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

# Usage:
#   ./play.sh --standing | --walking [--roboland]
#   ./play.sh --checkpoint /path/to/run_or_checkpoint [--epoch EPOCH] [--roboland]

POLICY=""
ROBOLAND=false
CHECKPOINT_ARG=""
EPOCH_ARG=""

for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    case "$arg" in
        --standing) POLICY="standing" ;;
        --walking)  POLICY="walking" ;;
        --roboland) ROBOLAND=true ;;
        --checkpoint)
            j=$((i+1))
            if [ $j -gt $# ]; then
                echo "ERROR: --checkpoint requires a path"
                exit 1
            fi
            CHECKPOINT_ARG="${!j}"
            i=$j
            ;;
        --epoch)
            j=$((i+1))
            if [ $j -gt $# ]; then
                echo "ERROR: --epoch requires a value"
                exit 1
            fi
            EPOCH_ARG="${!j}"
            i=$j
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 --standing | --walking [--roboland]"
            echo "   or: $0 --checkpoint /path/to/run_or_checkpoint [--epoch EPOCH] [--roboland]"
            exit 1
            ;;
    esac
done

# If no explicit checkpoint is provided, use policies.cfg
if [ -z "$CHECKPOINT_ARG" ]; then
    if [ -z "$POLICY" ]; then
        echo "Usage: $0 --standing | --walking [--roboland]"
        echo "   or: $0 --checkpoint /path/to/run_or_checkpoint [--epoch EPOCH] [--roboland]"
        exit 1
    fi

    source policies.cfg

    if [ "$POLICY" = "standing" ]; then
        RUN_DIR="$STANDING_POLICY_CHECKPOINT"
        DEFAULT_EPOCH="$STANDING_POLICY_CHECKPOINT_EPOCH"
    else
        RUN_DIR="$WALKING_POLICY_CHECKPOINT"
        DEFAULT_EPOCH="$WALKING_POLICY_CHECKPOINT_EPOCH"
    fi

    LOG_DIR="logs/rl_games/rocket_direct/$RUN_DIR/nn"
else
    # Explicit checkpoint mode: skip policies.cfg entirely
    RUN_DIR="$CHECKPOINT_ARG"
    DEFAULT_EPOCH=""
    LOG_DIR="$CHECKPOINT_ARG"

    # If the checkpoint path is a file, use it directly
    if [ -f "$CHECKPOINT_ARG" ]; then
        CHECKPOINT="$CHECKPOINT_ARG"
    else
        # Otherwise assume it is a run directory or nn directory
        if [ -d "$CHECKPOINT_ARG/nn" ]; then
            LOG_DIR="$CHECKPOINT_ARG/nn"
        elif [ -d "$CHECKPOINT_ARG" ]; then
            LOG_DIR="$CHECKPOINT_ARG"
        fi
    fi
fi

# Resolve checkpoint path if not already set directly
if [ -z "$CHECKPOINT" ]; then
    EPOCH_TO_USE="$EPOCH_ARG"
    if [ -z "$EPOCH_TO_USE" ] || [ "$EPOCH_TO_USE" = "none" ]; then
        EPOCH_TO_USE="$DEFAULT_EPOCH"
    fi

    CHECKPOINT="$LOG_DIR/rocket_direct.pth"

    if [ -n "$EPOCH_TO_USE" ] && [ "$EPOCH_TO_USE" != "none" ]; then
        CHECKPOINT=$(ls "$LOG_DIR"/last_rocket_direct_ep_${EPOCH_TO_USE}_*.pth 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            echo "ERROR: No checkpoint found for epoch $EPOCH_TO_USE in $LOG_DIR"
            exit 1
        fi
    fi
fi

echo "Run dir:    $RUN_DIR"
echo "Checkpoint: $CHECKPOINT"

# Source user credentials
if [ -f ~/setup.sh ]; then
    source ~/setup.sh
else
    echo "WARNING: ~/setup.sh not found! Credentials will not be loaded"
fi

if [ "$ROBOLAND" = true ]; then
    source ~/env_isaaclab/bin/activate
    python3 -u scripts/rl_games/play.py \
        --checkpoint "$CHECKPOINT" \
        --num_envs 1
else
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
        "$SIF_PATH" \
        bash -lc "
            /isaac-sim/python.sh -m pip install --user --no-build-isolation 'isaaclab[all]==2.3.2' --extra-index-url https://pypi.nvidia.com &&
            /isaac-sim/python.sh -m pip install --user --force-reinstall -e source/rocket &&
            /isaac-sim/python.sh -m pip install --user rl-games &&
            /isaac-sim/python.sh -m pip install --user imageio imageio-ffmpeg &&
            /isaac-sim/python.sh -u scripts/list_envs.py &&
            /isaac-sim/python.sh -u scripts/rl_games/play.py --num_envs 32 --headless --video --checkpoint '$CHECKPOINT'
        "
fi

echo "Job completed"
