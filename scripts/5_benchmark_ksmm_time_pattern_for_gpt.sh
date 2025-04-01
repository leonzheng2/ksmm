# Compare kernel vs bmm on the patterns

for pattern in "1,64,768,24" "96,192,64,1" "1,64,192,96" "24,768,64,1" "1,96,288,64" "48,384,32,1" "1,32,384,48" "64,288,96,1" "1,96,1152,16" "192,96,32,1" "1,32,96,192" "16,1152,96,1" "1,8,96,192" "48,384,128,1" "1,128,384,48" "192,96,8,1" "1,48,512,32" "64,256,96,1" "1,96,256,64" "32,512,48,1"
do
  for algo in "kernel" "bmm"
  do
    name="${algo}-${pattern}"
    python src/ksmm_py/benchmark/ksmm_time.py \
        --saving-csv "results/gpt-linear-layers/${name}.csv" \
        --device-id 0 \
        --patterns $pattern \
        --algo $algo \
        --bs-last 0 \
        --device cuda \
        --precision fp32 \
        --batch-size 25088
  done
done
