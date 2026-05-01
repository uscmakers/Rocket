#!/usr/bin/env bash
set -euo pipefail

policy_path=""
ROBOLAND=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      policy_path="$2"
      shift 2
      ;;
    --roboland)
      ROBOLAND=true
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$policy_path" ]]; then
  echo "Usage: ./export.sh --policy /path/to/rocket_direct.pth [--roboland]" >&2
  exit 1
fi

if [[ "$ROBOLAND" = true ]]; then
  source ~/env_isaaclab/bin/activate
  python3 -u scripts/rl_games/exporter.py --policy "$policy_path"
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
          /isaac-sim/python.sh -u scripts/rl_games/exporter.py --policy '$policy_path'
      "
fi
