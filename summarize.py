import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", default=False, action='store_true')
    parser.add_argument(
        "--cvss_col",
        choices=["access_vector", "access_complexity", "authentication", "confidentiality",
                 "integrity", "availability", "severity"],
        default=""
    )
    parser.add_argument("--arch", choices=["ffnn", "cnn", "lstm"], default="")
    parser.add_argument("--optimizer", type=str, default="")
    parser.add_argument("--early_stopping_metrics", choices=["f1", "mcc"], default="")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))

    root_dir = "./output"
    # runs = os.listdir(root_dir)
    runs = glob.glob(
        "*",
        root_dir=root_dir
    )
    print(f"Total {len(runs)} runs.")

    if not args.all:
        runs = glob.glob(
            f"*{args.cvss_col}**{args.arch}**{args.early_stopping_metrics}*",
            root_dir=root_dir
        )
    print(f"Found {len(runs)} runs.")
    # print(runs)
    print("test_file_name".ljust(35), "\taccuracy,precision,recall,f1,mcc")
    for run in runs:
        test_file = f"{root_dir}/{run}/test.csv"
        test = pd.read_csv(test_file)
        if len(test) != 1:
            print(f"{test_file} is invalid.")
            continue
        metrics = test.iloc[0].to_dict()
        print(f"{run.ljust(35)}\t{list(metrics.values())}")


if __name__ == '__main__':
    main()
