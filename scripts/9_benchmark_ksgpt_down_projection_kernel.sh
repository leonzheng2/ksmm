
# gpt2-large
model_id="gpt2-medium"
list_chains="results/0_ksmm/processed/interesting-gpt-1024x4096.txt"
saving_dir="results/3_gpt/9_benchmark_ksgpt_down_projection_kernel"

# dense
python3 src/ksmm_py/benchmark/ksgpt_time.py \
      --model-id $model_id \
      --batch-size 128 \
      --seq-length 196 \
      --precision "fp32" \
      --algo "dense" \
      --device "cuda" \
      --saving-dir $saving_dir

# KS linear
while read -r line_down; do
    echo "-----------------"
    read da1 db1 dc1 dd1 da2 db2 dc2 dd2 <<< $line_down
    echo "$da1 $db1 $dc1 $dd1 $da2 $db2 $dc2 $dd2"
    for algo in "kernel" "bmm"
    do
      python3 src/ksmm_py/benchmark/ksgpt_time.py \
        --model-id $model_id \
        --down-pattern1 $da1 $db1 $dc1 $dd1 \
        --down-pattern2 $da2 $db2 $dc2 $dd2 \
        --batch-size 128 \
        --seq-length 196 \
        --precision "fp32" \
        --algo $algo \
        --device "cuda" \
        --saving-dir $saving_dir
    done
done < $list_chains
