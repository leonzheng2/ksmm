import argparse
from pathlib import Path
import glob
import os
import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default="../raw")
    parser.add_argument("--ratio-rewrite-multiply", type=float)
    parser.add_argument("--save-dir", type=Path, default="../processed")
    return parser.parse_args()


def combined_df(directory):
    csv_files = glob.glob(os.path.join(str(directory), "*.csv"))
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
        except pd.errors.ParserError:
            print(f"Skipping file with parsing error: {file}")
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):
    if 'pattern' in df.columns:
        try:
            df['pattern'] = df['pattern'].apply(eval)
            df['input_size'] = df['pattern'].apply(
                lambda x: x[0][0] * x[0][2] * x[0][3] if isinstance(x, list) and len(x) > 0 and isinstance(x[0],
                                                                                                           list) and len(
                    x[0]) == 4 else None)
            df['output_size'] = df['pattern'].apply(
                lambda x: x[-1][0] * x[-1][1] * x[-1][3] if isinstance(x, list) and len(x) > 0 and isinstance(x[-1],
                                                                                                              list) and len(
                    x[-1]) == 4 else None)
        except (ValueError, IndexError, TypeError):
            print(
                "Error processing 'pattern'. Check if the column exists and is in the correct format (list of lists of four integers).")
    else:
        print("The 'pattern' column does not exist in the DataFrame.")

    df[['a1', 'b1', 'c1', 'd1', 'a2', 'b2', 'c2', 'd2']] = df['pattern'].apply(lambda x: pd.Series(parse_pattern(x)))
    df['size'] = df.apply(lambda row: (row['output_size'], row['input_size']), axis=1)
    df['hidden_dim'] = df["a1"] * df["c1"] * df["d1"]
    df["nnz"] = df["a1"] * df["b1"] * df["c1"] * df["d1"] + df["a2"] * df["b2"] * df["c2"] * df["d2"]
    df['ratio-rewrite-multiply'] = (df["a1"] * (df["b1"] + df["c1"]) * df["d1"] + df["a2"] * (df["b2"] + df["c2"]) * df[
        "d2"]) / df["nnz"]
    df["ratio-nnz-mn"] = df["nnz"] / (df['input_size'] * df['output_size'])
    return df


def list_to_string(nested_list):
    """Converts a list of lists to a space-separated string of comma-separated strings.

    Args:
      nested_list: A list of lists, where each inner list contains numbers.

    Returns:
      A string representation of the nested list. Returns an empty string if the input is invalid.
    """
    try:
        return " ".join([",".join(map(str, sublist)) for sublist in nested_list])
    except (TypeError, AttributeError):
        return ""


def write_strings_to_file(string_list, filepath):
    """Writes a list of strings to a text file, each string on a new line.

    Args:
      string_list: A list of strings to write to the file.
      filepath: The path to the output text file.
    """
    try:
        with open(filepath, 'w') as file:
            for string in string_list:
                file.write(string + '\n')
        print(f"Strings written to {filepath} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def parse_pattern(pattern):
    try:
        if isinstance(pattern, list) and len(pattern) == 2 and all(
                isinstance(sublist, list) and len(sublist) == 4 for sublist in pattern):
            a2, b2, c2, d2 = pattern[0]
            a1, b1, c1, d1 = pattern[1]
            return a1, b1, c1, d1, a2, b2, c2, d2
        else:
            return None, None, None, None, None, None, None, None
    except (ValueError, IndexError, TypeError):
        return None, None, None, None, None, None, None, None


if __name__ == "__main__":
    args = get_arguments()
    df = combined_df(args.raw_dir)
    df = preprocess(df)


    args.save_dir.mkdir(parents=True, exist_ok=True)

    size_list = df['size'].unique()
    for size in size_list:
        df_size = df[df["size"] == size]
        df_nn_linear = df_size[df_size["algo"] == "nn_linear"]
        assert len(df_nn_linear) == 1
        nn_linear_time = df_nn_linear['mean'].iloc[0]
        df_size["relative-time"] = df_size["mean"] / nn_linear_time
        df_filter = df_size[df_size["relative-time"] < 1]
        if args.ratio_rewrite_multiply is not None:
            print(f"Keeping only chains where ratio-rewrite-multiply > {args.ratio_rewrite_multiply}")
            df_filter = df_filter[df_filter["ratio-rewrite-multiply"] > args.ratio_rewrite_multiply]
        print(size, len(df_filter))
        df_filter = df_filter.sort_values(by=["ratio-rewrite-multiply"], ascending=False)
        name = f"{size[0]}x{size[1]}-ratio_rewrite_multiply={args.ratio_rewrite_multiply}"
        df_filter.to_csv(args.save_dir / f"{name}.csv")
        df_filter['pattern_string'] = df_filter['pattern'].apply(list_to_string)
        write_strings_to_file(df_filter['pattern_string'], args.save_dir / f"{name}.txt")
