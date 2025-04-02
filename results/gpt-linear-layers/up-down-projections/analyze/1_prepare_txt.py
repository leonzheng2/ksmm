import itertools
from pathlib import Path


def cartesian_product_and_partition(file1, file2, output_prefix):
    # Read the strings from the first and second files
    with open(file1, 'r') as f1:
        file1_lines = [line.strip() for line in f1.readlines()]

    with open(file2, 'r') as f2:
        file2_lines = [line.strip() for line in f2.readlines()]

    # Create the Cartesian product of the lines from both files
    product = itertools.product(file1_lines, file2_lines)

    # Prepare the resulting rows with "|" separator
    result_rows = [f"{item1}|{item2}" for item1, item2 in product]

    # Partition the rows into 8 files
    partitions = [[] for _ in range(8)]

    for i, row in enumerate(result_rows):
        partitions[i % 8].append(row)

    # Write the partitions to 8 separate files
    for i in range(8):
        with open(f"{output_prefix}-{i}.txt", 'w') as output_file:
            output_file.write("\n".join(partitions[i]))


if __name__ == "__main__":
    # Example usage
    file1 = Path('../processed/6144x1536-ratio_rewrite_multiply=0.015.txt')
    file2 = Path('../processed/1536x6144-ratio_rewrite_multiply=0.015.txt')
    output_prefix = '../processed/cartesian_up_down-1536-6144-ratio_rewrite_multiply=0.015'

    cartesian_product_and_partition(file1, file2, output_prefix)
