#!/bin/bash

# 用法:
# ./train_model.sh MODEL_TYPE [任意参数...]
# 例如:
# ./train_model.sh equigiga --dataset /path/to/new --dataset_raw /path/to/raw --epochs 50 --lr 0.001

MODEL=$1
shift 1  
EXTRA_ARGS="$@"

if [ -z "$MODEL" ]; then
  echo "Usage: $0 {equi_giga|equi_igd|giga|igd} [extra args...]"
  echo "e.g.: $0 equi_giga --dataset /new --dataset_raw /raw --epochs 12 --num_workers 4 --batch_size 128"
  exit 1
fi

case "$MODEL" in
  equi_giga)
    python train_equigiga.py $EXTRA_ARGS
    ;;
  equi_igd)
    python train_equigigd.py $EXTRA_ARGS
    ;;
  giga)
    python train_giga.py $EXTRA_ARGS
    ;;
  igd)
    python train_igd.py $EXTRA_ARGS
    ;;
  *)
    echo "Invalid model type: $MODEL"
    echo "Available types: equi_giga, equi_igd, giga, igd"
    exit 1
    ;;
esac
