#!/bin/bash

ratio_rewrite_multiply=0.015

source venv/bin/activate
python src/ksmm_py/benchmark/ksgpt_time.py --algo dense

tmux new-session -d -s "my-session"
tmux send-keys -t my-session:0 "bash scripts/8bis_process_patterns.sh results/gpt-linear-layers/up-down-projections/processed/cartesian_up_down-1536-6144-ratio_rewrite_multiply=${ratio_rewrite_multiply}-0.txt bmm 0" C-m
for i in {1..7}; do
  tmux new-window -t my-session:$i
  tmux send-keys -t my-session:$i "bash scripts/8bis_process_patterns.sh results/gpt-linear-layers/up-down-projections/processed/cartesian_up_down-1536-6144-ratio_rewrite_multiply=${ratio_rewrite_multiply}-${i}.txt bmm ${i}" C-m
done
tmux attach-session -t my-session
