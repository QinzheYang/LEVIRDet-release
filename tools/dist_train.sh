#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# A fixed default port makes independent jobs on the same host share the same
# rendezvous endpoint. Use an isolated, automatically assigned endpoint for
# normal single-node jobs. Setting PORT explicitly preserves static launch
# behavior, and multi-node jobs still use a fixed endpoint by necessity.
if [[ "$NNODES" == "1" && -z "${PORT:-}" ]]; then
    PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" \
    python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$GPUS" \
        "$SCRIPT_DIR/train.py" \
        "$CONFIG" \
        --launcher pytorch "${@:3}"
else
    PORT=${PORT:-29500}
    PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" \
    python -m torch.distributed.run \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$PORT" \
        --nproc_per_node="$GPUS" \
        "$SCRIPT_DIR/train.py" \
        "$CONFIG" \
        --launcher pytorch "${@:3}"
fi
