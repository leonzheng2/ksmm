#!/bin/bash

algo="kernel"
python3 src/ksmm_py/benchmark/ksllama_time.py \
  --algo=$algo \
  --up-pattern1 1 16 32 896  \
  --up-pattern2 8 3584 512 1 \
  --down-pattern1 1 512 3584 8 \
  --down-pattern2 896 32 16 1