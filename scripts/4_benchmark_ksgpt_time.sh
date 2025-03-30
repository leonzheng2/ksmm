#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 src/ksmm_py/benchmark/ksgpt_time.py \
  --algo=$1 \