#!/usr/bin/env bash
set -euo pipefail

policy_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      policy_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$policy_path" ]]; then
  echo "Usage: ./export.sh --policy /path/to/rocket_direct.pth" >&2
  exit 1
fi

python scripts/rl_games/exporter.py --policy "$policy_path"
