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
    parser.add_argument("--max", default=False, action='store_true')

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
    max_f1 = float("-inf")
    max_f1_run = None
    max_mcc = float("-inf")
    max_mcc_run = None
    for run in runs:
        test_file = f"{root_dir}/{run}/test.csv"
        try:
            test = pd.read_csv(test_file)
        except Exception as exc:
            print(exc)
            continue
        if len(test) != 1:
            print(f"{test_file} is invalid.")
            continue
        metrics = test.iloc[0].to_dict()
        # print(metrics)
        print(f"{run.ljust(35)}\t{list(metrics.values())}")
        if args.max:
            if metrics[" f1"] > max_f1:
                max_f1 = metrics[" f1"]
                max_f1_run = test_file
            if metrics[" mcc"] > max_mcc:
                max_mcc = metrics[" mcc"]
                max_mcc_run = test_file

    if max_f1_run:
        print("Max F1 Run")
        print(f"{max_f1_run.ljust(35)}\t{list(pd.read_csv(max_f1_run).iloc[0].to_dict().values())}")

    if max_mcc_run:
        print("Max MCC Run")
        print(f"{max_mcc_run.ljust(35)}\t{list(pd.read_csv(max_mcc_run).iloc[0].to_dict().values())}")


if __name__ == '__main__':
    main()
