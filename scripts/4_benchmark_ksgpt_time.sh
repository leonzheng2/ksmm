#!/bin/bash


# Read the up and down pairs into arrays (each line is one entry)
mapfile -t up_pairs < up_pairs.txt
mapfile -t down_pairs < down_pairs.txt


# Iterate over all combinations of up and down pairs
for up_line in "${up_pairs[@]}"; do
    # Split the line by the delimiter "|" into up-pattern1 and up-pattern2.
    IFS='|' read -r up_pattern1 up_pattern2 <<< "$up_line"
    # Trim any leading/trailing whitespace.
    up_pattern1=$(echo "$up_pattern1" | xargs)
    up_pattern2=$(echo "$up_pattern2" | xargs)
    
    for down_line in "${down_pairs[@]}"; do
        # Split the down line into down-pattern1 and down-pattern2.
        IFS='|' read -r down_pattern1 down_pattern2 <<< "$down_line"
        down_pattern1=$(echo "$down_pattern1" | xargs)
        down_pattern2=$(echo "$down_pattern2" | xargs)
        # Construct the command.
          for algo in bmm kernel; do
          cmd="CUDA_VISIBLE_DEVICES=1 python3 src/ksmm_py/benchmark/ksgpt_time.py \
            --algo=$algo \
            --up-pattern1 $up_pattern1 \
            --up-pattern2 $up_pattern2 \
            --down-pattern1 $down_pattern1 \
            --down-pattern2 $down_pattern2"
            
          # Print the command for verification.
          echo "Running: $cmd"
          
          # Run the command (remove 'echo' if you want to execute it)
          eval $cmd
      done
  done
done
