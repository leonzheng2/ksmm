
# gpt2-large
model_id="gpt2-medium"
list_chains="results/0_ksmm/processed/interesting-gpt-1024x4096.txt"
saving_dir="results/3_gpt/9bis_check_one_kslinear_old"

# dense
python src/ksmm_py/benchmark/ksmm_time.py \
    --patterns "1,1024,4096,1" \
    --batch-size 25088 \
    --precision "fp32" \
    --bs-last 0 \
    --algo "nn_linear" \
    --device "cuda" \
    --saving-csv "${saving_dir}/dense.csv"

# KS linear
while read -r line_down; do
    echo "-----------------"
    read da1 db1 dc1 dd1 da2 db2 dc2 dd2 <<< $line_down
    echo "$da1 $db1 $dc1 $dd1 $da2 $db2 $dc2 $dd2"
    for algo in "kernel" "bmm"
    do
      python src/ksmm_py/benchmark/ksmm_time.py \
        --patterns "${da2},${db2},${dc2},${dd2}" "${da1},${db1},${dc1},${dd1}" \
        --batch-size 25088 \
        --precision "fp32" \
        --bs-last 0 \
        --algo $algo \
        --device "cuda" \
        --saving-csv "${saving_dir}/algo=${algo}-${da2},${db2},${dc2},${dd2}-${da1},${db1},${dc1},${dd1}.csv"
    done
done < $list_chains
