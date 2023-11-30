import multiprocessing
import os
from pathlib import Path


def get_data_location():
    if os.path.exists("./data"):
        data_folder = "./data"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1
):

    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())

