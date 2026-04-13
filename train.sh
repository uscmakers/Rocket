#!/bin/bash
#SBATCH --account=biyik_1173
#SBATCH --job-name=rocket-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100"
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=b17-15
#SBATCH --output=jobs/rocket-train_%j.out
#SBATCH --error=jobs/rocket-train_%j.err

# Usage: ./train.sh [--roboland] [extra args passed to train.py, e.g. --standing --walking --checkpoint ...]

# Parse arguments
ROBOLAND=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --roboland) ROBOLAND=true; shift ;;
        *) EXTRA_ARGS+=("$1");     shift ;;
    esac
done

# Source user credentials from setup.sh
if [ -f ~/setup.sh ]; then
    source ~/setup.sh
else
    echo "WARNING: ~/setup.sh not found! Credentials will not be loaded"
fi

TRAIN_ARGS=(
    --max_iterations 10000

    --track
    --wandb-entity rocket-babysitters
    --wandb-project-name rocket
    "${EXTRA_ARGS[@]}"
)

if [ "$ROBOLAND" = true ]; then
    # Run locally with custom conda/venv environment
    source ~/env_isaaclab/bin/activate
    python3 -u scripts/rl_games/train.py "${TRAIN_ARGS[@]}"
else
    # Set environment variables
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y

    # Path to SIF file
    SIF_PATH="$HOME/isaac-sim_5.1.0.sif"

    # Load Apptainer
    module load apptainer
    apptainer --version

    mkdir -p ~/isaac-sim-cache/data
    mkdir -p ~/isaac-sim-cache/cache
    mkdir -p ~/isaac-sim-cache/logs

    # Run training with Apptainer
    apptainer exec --nv \
        --bind ~/isaac-sim-cache/data:/isaac-sim/kit/data \
        --bind ~/isaac-sim-cache/cache:/isaac-sim/kit/cache \
        --bind ~/isaac-sim-cache/logs:/isaac-sim/kit/logs \
        $SIF_PATH \
        bash -c "
        /isaac-sim/python.sh -m pip install --user --no-build-isolation 'isaaclab[all]==2.3.2' --extra-index-url https://pypi.nvidia.com &&
        /isaac-sim/python.sh -m pip install --user --force-reinstall -e source/rocket &&
        /isaac-sim/python.sh -m pip install --user rl-games &&
        /isaac-sim/python.sh -m pip install --user imageio imageio-ffmpeg &&
        /isaac-sim/python.sh -u scripts/list_envs.py &&
        /isaac-sim/python.sh -u scripts/rl_games/train.py "${TRAIN_ARGS[@]}"
        "
fi

echo "Job completed"
