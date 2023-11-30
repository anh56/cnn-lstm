from io import BytesIO
import urllib.request
from zipfile import ZipFile
import os
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

# Let's see if we have an available GPU
import numpy as np
import random


def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("data"):
        data_folder = "data"
    else:
        raise IOError("Make sure dataset exist")

    return data_folder


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) == 1:
        precision = accuracy_score(y_true, y_pred)
        recall = precision
        f1 = precision
    elif len(np.unique(np.r_[y_true, y_pred])) == 2:

        precisions = []
        recalls = []

        for label in np.unique(np.r_[y_true, y_pred]):
            precisions.append(precision_score(y_true, y_pred, average='binary', pos_label=label))
            recalls.append(recall_score(y_true, y_pred, average='binary', pos_label=label))

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = (2 * precision * recall) / (precision + recall)

    else:
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, precision, recall, f1, mcc


def save_metrics(accuracy, precision, recall, f1, mcc, result_save_path):
    if not os.path.exists(result_save_path):
        outfile = open(result_save_path, 'w')
        outfile.write("accuracy, precision, recall, f1, mcc\n")
    else:
        outfile = open(result_save_path, 'a')
    output = (
        f"{round(accuracy, 4)},{round(precision, 4)},{round(recall, 4)},"
        f"{round(f1, 4)},{round(mcc, 4)}\n"
    )
    outfile.write(output)
    outfile.close()


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])


def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)
