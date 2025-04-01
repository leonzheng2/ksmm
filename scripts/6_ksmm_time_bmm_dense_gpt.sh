#!/bin/bash

# emb_dim
emb_dim=1536

# nn_linear
algo="nn_linear"
for pattern in "1,${emb_dim},$((4 * emb_dim)),1" "1,$((4 * emb_dim)),${emb_dim},1"
do
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
done

# KS linear
process_patterns() {
  local candidates=$1
  local algo=$2

  while read -r line; do
    read a1 b1 c1 d1 a2 b2 c2 d2 <<< $line
    echo "$a1 $b1 $c1 $d1 $a2 $b2 $c2 $d2"
    pattern1="${a1},${b1},${c1},${d1}"
    pattern2="${a2},${b2},${c2},${d2}"
    name="${algo}-${pattern1}-${pattern2}"
    python src/ksmm_py/benchmark/ksmm_time.py \
        --saving-csv "results/gpt-linear-layers/up-down-projections/${name}.csv" \
        --device-id 0 \
        --patterns $pattern1 $pattern2 \
        --algo $algo \
        --bs-last 0 \
        --device cuda \
        --precision fp32 \
        --batch-size 25088
  done < $candidates
}

process_patterns "scripts/gpt-patterns/candidates-${emb_dim}-up.txt" "bmm"
process_patterns "scripts/gpt-patterns/candidates-${emb_dim}-down.txt" "bmm"

# C'est un premier filtre. Après il faudra vérifier que ça marche aussi dans le ViT complet.