#!/bin/bash

MODEL=$1
shift 1
EXTRA_ARGS="$@"

if [ -z "$MODEL" ]; then
  echo "Usage: $0 {equi_giga|equi_igd} [extra args...]"
  echo "e.g.: $0 equi_giga --dataset /new --dataset_raw /raw --epochs 12 --num_workers 4 --batch_size 128"
  exit 1
fi

case "$MODEL" in
  equi_giga)
    python train_equigiga.py $EXTRA_ARGS
    ;;
  equi_igd)
    python train_equiigd.py $EXTRA_ARGS
    ;;
  *)
    echo "Invalid model type: $MODEL"
    echo "Available types: equi_giga, equi_igd"
    exit 1
    ;;
esac
