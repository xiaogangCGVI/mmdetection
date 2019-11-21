#!/usr/bin/env bash

 export KMP_INIT_AT_FORK=FALSE
PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

echo $(dirname "$0")
echo ${@:3}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
