#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/config.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Configuration file does not exist: $CONFIG_PATH" >&2
    exit 1
fi
CONFIG_PATH="$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)/$(basename -- "$CONFIG_PATH")"
MODEL="$(awk '$1 == "model:" { print $2; exit }' "$CONFIG_PATH")"
if [[ -z "$MODEL" ]]; then
    echo "Could not find a model setting in $CONFIG_PATH" >&2
    exit 1
fi
MODEL_COUNT="$(awk '
    /^MODEL_REGISTRY:/ { in_registry = 1; next }
    in_registry && /^}/ { print count; exit }
    in_registry && /^    "[^"]+":/ { count++ }
' "$SCRIPT_DIR/model_factory.py")"
if [[ ! "$MODEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Could not determine the available model count from model_factory.py" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
mkdir -p logs
if [[ "$MODEL" == "-1" ]]; then
    LAST_MODEL_INDEX=$((MODEL_COUNT - 1))
    sbatch --array="0-$LAST_MODEL_INDEX%4" "$SCRIPT_DIR/slurm_trainig.slurm" \
        "$SCRIPT_DIR" "$CONFIG_PATH"
else
    sbatch "$SCRIPT_DIR/slurm_trainig.slurm" "$SCRIPT_DIR" "$CONFIG_PATH"
fi
