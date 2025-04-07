import os
import pandas as pd
import argparse
from pathlib import Path
import ast


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default="../raw-rtx6000")
    parser.add_argument("--processed-dir", type=Path, default="../processed")
    return parser.parse_args()


def concatenate_csvs(directory_path, output_filename="concatenated.csv"):
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Read and concatenate them
    dataframes = []
    for file in csv_files:
        full_path = os.path.join(directory_path, file)
        df = pd.read_csv(full_path)
        dataframes.append(df)

    if not dataframes:
        print("No CSV files found.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


if __name__ == "__main__":
    args = get_arguments()
    df = concatenate_csvs(args.raw_dir)
    df_dense = df[df["algo"] == "dense"]
    assert len(df_dense) == 1
    dense_time = df_dense["mean"].iloc[0]
    dense_num_params = df_dense["number_parameters"].iloc[0]
    df["time / dense_time"] = df["mean"] / dense_time
    df["density"] = df["number_parameters"] / dense_num_params
    df = df.sort_values(by=["up-pattern1", "up-pattern2", "down-pattern1", "down-pattern2", "algo"])
    print(df)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.processed_dir / "combined.csv")


    # df["time / dense_time"] = df["mean"] / dense_time
    # df = df[df["time / dense_time"] < 1]
    # df = df.sort_values(by="time / dense_time")
    # print(df)
    #
    # columns = ["up-pattern1", "up-pattern2", "down-pattern1", "down-pattern2"]
    # for column in columns:
    #     df[column] = df[column].apply(ast.literal_eval)
    #     df[column] = df[column].apply(lambda x: ",".join(map(str, x)))
    #
    # series_list = [df[column] for column in columns]
    # result_set = set().union(*series_list)
    # print(result_set)
    #
    # args.processed_dir.mkdir(parents=True, exist_ok=True)
    # with open(args.processed_dir / "patterns-to-finetune.txt", "w") as f:
    #     for item in result_set:
    #         f.write(item + "\n")