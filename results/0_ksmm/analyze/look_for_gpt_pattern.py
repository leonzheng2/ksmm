import pandas as pd
import argparse
from pathlib import Path
import ast


def get_arguments():
    emb_dim = 384

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default="./results/0_ksmm_time.csv")
    parser.add_argument("--input-dim", type=int, default=emb_dim)
    parser.add_argument("--output-dim", type=int, default=4 * emb_dim)
    parser.add_argument("--bs-position", type=str, default="bs_first")
    parser.add_argument("--batch-size", type=int, default=25088)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--algo1", type=str, default="kernel")
    parser.add_argument("--algo2", type=str, default="bmm")
    parser.add_argument("--save-dir", type=Path, default="./results/0_ksmm/processed")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    df = pd.read_csv(args.raw_path)
    df['patterns'] = df['patterns'].apply(ast.literal_eval)
    df[['a', 'b', 'c', 'd']] = pd.DataFrame(df['patterns'].apply(lambda x: x[0]).tolist(), index=df.index)
    print(df.head())

    df = df[df["bs_position"] == args.bs_position]
    df = df[df["precision"] == args.precision]
    df = df[df["batch-size"] == args.batch_size]
    df = df[df['algo'].isin([args.algo1, args.algo2])]
    df = df[['a', 'b', 'c', 'd', 'algo', 'mean']]
    df = df.rename(columns={"mean": "time"})
    df["input_dim"] = df["a"] * df["c"] * df["d"]
    df["output_dim"] = df["a"] * df["b"] * df["d"]
    print(df.columns)

    df_algo1 = df[df['algo'] == args.algo1]
    df_algo2 = df[df['algo'] == args.algo2]
    df = pd.merge(df_algo1, df_algo2, on=['a', 'b', 'c', 'd', 'input_dim', 'output_dim'],
                  suffixes=(f'_{args.algo1}', f'_{args.algo2}'))


    def reduce_to_pattern(_df):
        _df = _df[["a", "b", "c", "d", "input_dim", "output_dim", f"time_{args.algo1}", f"time_{args.algo2}"]]
        _df = _df.drop_duplicates()
        return _df


    # up
    df_pattern1 = df[df["output_dim"] == args.output_dim]
    df_pattern1 = df_pattern1[df_pattern1["a"] == 1]
    df_pattern1 = reduce_to_pattern(df_pattern1)

    # down
    df_pattern2 = df[df["input_dim"] == args.input_dim]
    df_pattern2 = df_pattern2[df_pattern2["d"] == 1]
    df_pattern2 = reduce_to_pattern(df_pattern2)

    result = pd.merge(df_pattern1, df_pattern2, left_on='input_dim', right_on='output_dim', suffixes=('_1', '_2'))
    result[f"time_{args.algo1}"] = result[f"time_{args.algo1}_1"] + result[f"time_{args.algo1}_2"]
    result[f"time_{args.algo2}"] = result[f"time_{args.algo2}_1"] + result[f"time_{args.algo2}_2"]
    result["ratio_1"] = result[f"time_{args.algo1}_1"] / result[f"time_{args.algo2}_1"]
    result["ratio_2"] = result[f"time_{args.algo1}_2"] / result[f"time_{args.algo2}_2"]
    result["ratio"] = result[f"time_{args.algo1}"] / result[f"time_{args.algo2}"]
    result = result[['a_1', 'b_1', 'c_1', 'd_1', 'a_2', 'b_2', 'c_2', 'd_2', 'ratio_1', 'ratio_2', 'ratio']]
    args.save_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.save_dir / f"valid-gpt-{args.output_dim}x{args.input_dim}.csv")

    interesting = result[result["ratio"] < 1]
    print(interesting)
    interesting.to_csv(args.save_dir / f"interesting-gpt-{args.output_dim}x{args.input_dim}.csv")

