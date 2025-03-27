#!/bin/bash

algo="bmm"

while read -r line_up; do
    while read -r line_down; do
        echo "-----------------"
        read ua1 ub1 uc1 ud1 ua2 ub2 uc2 ud2 <<< $line_up
        echo "$ua1 $ub1 $uc1 $ud1 $ua2 $ub2 $uc2 $ud2"
        read da1 db1 dc1 dd1 da2 db2 dc2 dd2 <<< $line_down
        echo "$da1 $db1 $dc1 $dd1 $da2 $db2 $dc2 $dd2"
        python3 src/ksmm_py/benchmark/ksllama_time.py \
          --algo=$algo \
          --up-pattern1 $ua1 $ub1 $uc1 $ud1 \
          --up-pattern2 $ua2 $ub2 $uc2 $ud2 \
          --down-pattern1 $da1 $db1 $dc1 $dd1 \
          --down-pattern2 $da2 $db2 $dc2 $dd2
    done < scripts/llama-patterns/candidates_4096_14336.txt
done < scripts/llama-patterns/candidates_14336_4096.txt

