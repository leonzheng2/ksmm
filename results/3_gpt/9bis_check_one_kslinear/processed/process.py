import os
import pandas as pd


def concatenate_csvs(directory_path):
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
    df = concatenate_csvs("..")
    df = df.sort_values(by=["pattern", "algo"])
    df.to_csv("combined.csv")

