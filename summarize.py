import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

cols = ["access_vector", "access_complexity", "authentication", "confidentiality",
        "integrity", "availability", "severity"]
archs = ["ffnn", "cnn", "lstm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", default=False, action='store_true')
    parser.add_argument(
        "--cvss_col",
        choices=cols,
        default=""
    )
    parser.add_argument("-a", "--arch", choices=archs, default="")
    parser.add_argument("-opt", "--optimizer", type=str, default="")
    parser.add_argument("-esm", "--early_stopping_metrics", choices=["f1", "mcc"], default="")
    parser.add_argument("-m", "--max", default=False, action='store_true')
    parser.add_argument("-gba", "--group_by_arch", default=False, action='store_true')

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
    print(runs)
    # print("\ntest_file_name".ljust(35), "\taccuracy,precision,recall,f1,mcc")
    print("\ntest_file_name,col,accuracy,precision,recall,f1,mcc")
    max_f1 = float("-inf")
    max_f1_run = None
    max_mcc = float("-inf")
    max_mcc_run = None

    errors = []
    for run in runs:
        test_file = f"{root_dir}/{run}/test.csv"
        cvss = ""
        for col in cols:
            if col in run:
                cvss = col
        try:
            test = pd.read_csv(test_file)
        except Exception as exc:
            # print(exc)
            errors.append(run)
            continue
        if len(test) != 1:
            print(f"{test_file} is invalid.")
            continue
        metrics = test.iloc[0].to_dict()
        # print(metrics)
        print(run, cvss, *list(metrics.values()), sep=",")
        if args.max:
            if metrics[" f1"] > max_f1:
                max_f1 = metrics[" f1"]
                max_f1_run = test_file
            if metrics[" mcc"] > max_mcc:
                max_mcc = metrics[" mcc"]
                max_mcc_run = test_file

    if max_f1_run:
        print("\nMax F1 Run")
        print(f"{max_f1_run.ljust(35)}\t{list(pd.read_csv(max_f1_run).iloc[0].to_dict().values())}")

    if max_mcc_run:
        print("\nMax MCC Run")
        print(f"{max_mcc_run.ljust(35)}\t{list(pd.read_csv(max_mcc_run).iloc[0].to_dict().values())}")

    if errors:
        print("\nError runs")
        for err in errors:
            print(err)


if __name__ == '__main__':
    main()
