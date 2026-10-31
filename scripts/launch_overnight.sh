#!/usr/bin/env bash
# Launch an overnight queue under nohup with the real engine.
#   scripts/launch_overnight.sh configs/overnight_a.yaml
# Stop it politely after the current entry: touch ~/GOTRAIN_STOP
set -euo pipefail
plan="${1:?usage: launch_overnight.sh <plan.yaml>}"
cd "$(dirname "$0")/.."

if command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$busy" ]; then
    echo "refusing: a process is already using the GPU (pid: $(echo "$busy" | tr '\n' ' '))" >&2
    exit 1
  fi
fi
if [ -e "$HOME/GOTRAIN_STOP" ]; then
  echo "refusing: $HOME/GOTRAIN_STOP exists (remove it first)" >&2
  exit 1
fi

mkdir -p "$HOME/gotrain_home"
log="$HOME/gotrain_home/overnight-$(basename "$plan" .yaml)-$(date -u +%Y%m%d-%H%M%S).log"
GOTRAIN_ENGINE=real GOTRAIN_HOME="${GOTRAIN_HOME:-$HOME/gotrain_home}" \
  nohup python scripts/overnight.py "$plan" > "$log" 2>&1 &
echo "PID: $!"
echo "Log: $log"
