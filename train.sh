#!/bin/bash

# Default to 1 node and 2 processes per node for training
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

echo "Starting training with ${NNODES} node(s) and ${NPROC_PER_NODE} process(es) per node..."

torchrun \
    --nnodes ${NNODES} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --rdzv_id 18635 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29502 \
    train.py --config configs/HumanGS_config.yaml