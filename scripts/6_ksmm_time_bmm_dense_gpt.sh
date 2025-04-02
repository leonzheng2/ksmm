#!/bin/bash

# emb_dim
emb_dim=$1
projection=$2

# nn_linear
algo="nn_linear"
if [ "$projection" == "up" ]; then
    pattern="1,$((4 * emb_dim)),${emb_dim},1"
elif [ "$projection" == "down" ]; then
    pattern="1,${emb_dim},$((4 * emb_dim)),1"
fi
name="${algo}-${pattern}"
python src/ksmm_py/benchmark/ksmm_time.py \
    --saving-csv "results/gpt-linear-layers/up-down-projections/${name}.csv" \
    --device-id 0 \
    --patterns $pattern \
    --algo $algo \
    --bs-last 0 \
    --device cuda \
    --precision fp32 \
    --batch-size 25088

source venv/bin/activate
tmux new-session -d -s "my-session"
tmux send-keys -t my-session:0 "bash scripts/6bis_process_patterns.sh scripts/gpt-patterns/candidates-${emb_dim}-${projection}-0.txt bmm 0" C-m
for i in {1..7}; do
  tmux new-window -t my-session:$i
  tmux send-keys -t my-session:$i "bash scripts/6bis_process_patterns.sh scripts/gpt-patterns/candidates-${emb_dim}-${projection}-${i}.txt bmm ${i}" C-m
done
tmux attach-session -t my-session
