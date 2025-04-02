candidates=$1
algo=$2
device_id=$3

# Activate virtual environment once before the loop
source venv/bin/activate

while IFS= read -r line; do
  # Use read with quotes to prevent splitting issues
  read a1 b1 c1 d1 a2 b2 c2 d2 <<< "$line"
  echo "$a1 $b1 $c1 $d1 $a2 $b2 $c2 $d2"
  pattern1="${a1},${b1},${c1},${d1}"
  pattern2="${a2},${b2},${c2},${d2}"
  name="${algo}-${pattern1}-${pattern2}"

  CUDA_VISIBLE_DEVICES=$device_id python src/ksmm_py/benchmark/ksmm_time.py \
      --saving-csv "results/gpt-linear-layers/up-down-projections/${name}.csv" \
      --device-id 0 \
      --patterns $pattern2 $pattern1 \
      --algo $algo \
      --bs-last 0 \
      --device cuda \
      --precision fp32 \
      --batch-size 25088
done < "$candidates"