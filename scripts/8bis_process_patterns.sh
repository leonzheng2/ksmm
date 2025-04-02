chain_list=$1
algo=$2
i=$3

while read -r line; do
    echo "-----------------"
    echo $line
    pattern_up=$(echo $line | cut -d'|' -f1)
    up_pattern2=$(echo $pattern_up | cut -d' ' -f1)
    up_pattern1=$(echo $pattern_up | cut -d' ' -f2)
    pattern_down=$(echo $line | cut -d'|' -f2)
    down_pattern2=$(echo $pattern_down | cut -d' ' -f1)
    down_pattern1=$(echo $pattern_down | cut -d' ' -f2)

    IFS=',' read -r ua1 ub1 uc1 ud1 <<< "$up_pattern1"
    IFS=',' read -r ua2 ub2 uc2 ud2 <<< "$up_pattern2"
    IFS=',' read -r da1 db1 dc1 dd1 <<< "$down_pattern1"
    IFS=',' read -r da2 db2 dc2 dd2 <<< "$down_pattern2"

    CUDA_VISIBLE_DEVICES=$i python src/ksmm_py/benchmark/ksgpt_time.py \
      --algo=$algo \
      --up-pattern1 $ua1 $ub1 $uc1 $ud1 \
      --up-pattern2 $ua2 $ub2 $uc2 $ud2 \
      --down-pattern1 $da1 $db1 $dc1 $dd1 \
      --down-pattern2 $da2 $db2 $dc2 $dd2
done < $chain_list
