import pandas as pd
import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dim", type=int, required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--ratio-rewrite-multiply", type=float)
    parser.add_argument("--density", type=float)
    parser.add_argument("--power-list", type=int, nargs='+')
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--comma-separator", action="store_true")
    return parser.parse_args()


def find_integers_1(m, n):
    result = set()

    # Iterate over possible values of b_1, c_1, d_1, a_2, b_2, c_2
    for b_1 in range(1, m + 1):
        for c_1 in range(1, n):
            for d_1 in range(2, m + 1):  # Eliminate diagonal cases
                for a_2 in range(2, n + 1):  # Eliminate dense cases
                    for b_2 in range(1, m):
                        for c_2 in range(1, n + 1):
                            # Check the conditions
                            if b_1 * d_1 == m and a_2 * c_2 == n and c_1 * d_1 == a_2 * b_2 and b_1 * c_1 * d_1 + a_2 * b_2 * c_2 < m * n and (
                                    b_1, c_1) != (1, 1) and (b_2, c_2) != (1, 1):
                                result.add(((1, b_1, c_1, d_1), (a_2, b_2, c_2, 1)))

    return result


def get_divisor_pairs(x):
    """
    Returns a list of all pairs (d1, d2) such that d1 * d2 = x.
    """
    pairs = []
    # To optimize, we can go only up to int(sqrt(x)) and find the complementary pair.
    # But for clarity, here we scan 1..x.
    for d1 in range(1, int(x ** 0.5) + 1):
        if x % d1 == 0:
            d2 = x // d1
            # (d1, d2)
            pairs.append((d1, d2))
            # If d1 != d2, also add (d2, d1) so we consider both orders
            if d1 != d2:
                pairs.append((d2, d1))
    return pairs


def find_integers_2(m, n):
    result = set()

    # 1) Get all (b_1, d_1) pairs such that b_1 * d_1 = m
    bd_pairs = get_divisor_pairs(m)

    # 2) Get all (a_2, c_2) pairs such that a_2 * c_2 = n
    ac_pairs = get_divisor_pairs(n)

    # 3) Combine factor pairs and search for b_2, c_1
    for (b_1, d_1) in bd_pairs:
        if d_1 > 1:
            for (a_2, c_2) in ac_pairs:
                if a_2 > 1:
                    # We want b_2 in [1, m), so check b_2 < m
                    for b_2 in range(1, m):
                        # We need c_1 = (a_2 * b_2) / d_1 to be an integer and c_1 < n
                        numerator = a_2 * b_2
                        if numerator % d_1 == 0:  # ensures c_1 is integer
                            c_1 = numerator // d_1
                            if c_1 < n:
                                # Check final condition:
                                lhs = b_1 * c_1 * d_1 + a_2 * b_2 * c_2  # LHS of the inequality
                                if lhs < m * n:
                                    if (b_1, c_1) != (1, 1) and (b_2, c_2) != (1, 1):
                                        result.add(((1, b_1, c_1, d_1), (a_2, b_2, c_2, 1)))

    return result


# selection
def ratio_rewritings_multiplication(pair):
    pattern_1, pattern_2 = pair
    a_1, b_1, c_1, d_1 = pattern_1
    a_2, b_2, c_2, d_2 = pattern_2
    return (a_1 * (b_1 + c_1) * d_1 + a_2 * (b_2 + c_2) * d_2) / (a_1 * b_1 * c_1 * d_1 + a_2 * b_2 * c_2 * d_2)


def ratio_nnz_mn(pair):
    pattern_1, pattern_2 = pair
    a_1, b_1, c_1, d_1 = pattern_1
    a_2, b_2, c_2, d_2 = pattern_2
    m = a_1 * b_1 * d_1
    n = a_2 * c_2 * d_2
    return (a_1 * b_1 * c_1 * d_1 + a_2 * b_2 * c_2 * d_2) / (m * n)


def convert_to_dataframe(lst):
    # Flatten each tuple and create the columns for the DataFrame
    columns = ['a_1', 'b_1', 'c_1', 'd_1', 'a_2', 'b_2', 'c_2', 'd_2']
    flattened_data = [(x1, x2, x3, x4, y1, y2, y3, y4) for (x1, x2, x3, x4), (y1, y2, y3, y4) in lst]
    df = pd.DataFrame(flattened_data, columns=columns)

    df["hidden_dim"] = df["a_1"] * df["c_1"] * df["d_1"]
    df['ratio_rewritings_multiplication'] = df.apply(
        lambda row: ratio_rewritings_multiplication((row[:4].values, row[4:8].values)), axis=1)
    df['ratio_nnz_mn'] = df.apply(lambda row: ratio_nnz_mn((row[:4].values, row[4:8].values)), axis=1)
    return df


def selection(
        original_df,
        hidden_dim_threshold=None,
        ratio_rewritings_multiplication_threshold=None,
        ratio_nnz_mn_threshold=None,
        powers_list=None,
):
    def is_divisible_by_power_of_2(num, min_power):
        """Checks if a number is divisible by at least 2^min_power."""
        if num == 0: return False  # handle the case 0
        return num % 2 ** min_power == 0

    df = original_df
    if hidden_dim_threshold is not None:
        df = df[df["hidden_dim"] <= hidden_dim_threshold]
    if ratio_rewritings_multiplication_threshold is not None:
        df = df[df["ratio_rewritings_multiplication"] > ratio_rewritings_multiplication_threshold]
    if ratio_nnz_mn_threshold is not None:
        df = df[df["ratio_nnz_mn"] >= ratio_nnz_mn_threshold]
    if powers_list is not None:
        assert len(powers_list) == 6
        mask = (
                df.apply(lambda row: is_divisible_by_power_of_2(row["b_1"], powers_list[0]), axis=1) &
                df.apply(lambda row: is_divisible_by_power_of_2(row["c_1"], powers_list[1]), axis=1) &
                df.apply(lambda row: is_divisible_by_power_of_2(row["d_1"], powers_list[2]), axis=1) &
                df.apply(lambda row: is_divisible_by_power_of_2(row["a_2"], powers_list[3]), axis=1) &
                df.apply(lambda row: is_divisible_by_power_of_2(row["b_2"], powers_list[4]), axis=1) &
                df.apply(lambda row: is_divisible_by_power_of_2(row["c_2"], powers_list[5]), axis=1)
        )
        df = df[mask]
    df = df.sort_values(by="ratio_rewritings_multiplication", ascending=False)
    return df


def generate_text_file(df, filename, comma_seperator):
    with open(filename, "w") as f:
        for index, row in df.iterrows():
            if comma_seperator:
                line = f"{int(row['a_1'])},{int(row['b_1'])},{int(row['c_1'])},{int(row['d_1'])} {int(row['a_2'])},{int(row['b_2'])},{int(row['c_2'])},{int(row['d_2'])}\n"
            else:
                line = f"{int(row['a_1'])} {int(row['b_1'])} {int(row['c_1'])} {int(row['d_1'])} {int(row['a_2'])} {int(row['b_2'])} {int(row['c_2'])} {int(row['d_2'])}\n"
            f.write(line)
    print(f"File '{filename}' created successfully.")


if __name__ == "__main__":
    args = get_arguments()
    result = find_integers_2(args.output_dim, args.input_dim)
    df = convert_to_dataframe(result)
    df = selection(
        df,
        args.hidden_dim,
        args.ratio_rewrite_multiply,
        args.density,
        args.power_list,
    )
    name=f"chains-{args.output_dim}x{args.input_dim}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_dir / f"{name}.csv")
    generate_text_file(df, args.save_dir / f"{name}.txt", args.comma_separator)
