import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pathlib import Path
import multiprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from .helpers import get_data_location
import matplotlib.pyplot as plt


class CodeDataset(Dataset):
    def __init__(self, x_vectorized, y_encoded):
        self.x_vectorized = x_vectorized
        self.y_encoded = y_encoded

    def __len__(self):
        return len(self.x_vectorized)

    def __getitem__(self, index):
        return self.x_vectorized[index], self.y_encoded[index]


def get_data(args):
    base_path = Path(get_data_location())
    input_file = base_path / args.input_file_name
    df = pd.read_csv(input_file)
    df = df[["func_before", args.cvss_col]].rename(
        columns={
            "func_before": "code",
            args.cvss_col: "category"
        }
    )
    print(df.shape, Counter(df["category"]).most_common())
    le = LabelEncoder().fit(df.category)
    df['category'] = le.transform(df['category'])
    print(df.shape, Counter(df["category"]).most_common())
    train_df, tmp_df = train_test_split(
        df,
        test_size=args.test_size + args.val_size,
        stratify=df.category,
        random_state=42,
    )
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=args.val_size / (args.test_size + args.val_size),
        stratify=tmp_df.category,
        random_state=42
    )
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)
    print("len(train_df)", len(train_df), "len(test_df)", len(test_df), "len(val_df)", len(val_df))
    return train_df, test_df, val_df


def gen_tok_pattern():
    single_toks = ['<=', '>=', '<', '>', '\\?', '\\/=', '\\+=', '\\-=', '\\+\\+', '--', '\\*=', '\\+', '-', '\\*',
                   '\\/', '!=', '==', '=', '!', '&=', '&', '\\%', '\\|\\|', '\\|=', '\\|', '\\$', '\\:']

    single_toks = '(?:' + '|'.join(single_toks) + ')'

    word_toks = '(?:[a-zA-Z0-9]+)'

    return single_toks + '|' + word_toks


def get_features_tf_idf(x_train, x_test, x_val):
    vectorizer = TfidfVectorizer(token_pattern=gen_tok_pattern())
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_val = vectorizer.transform(x_val)
    print("len(vectorizer.vocabulary_)", len(vectorizer.vocabulary_))

    x_train_dense = x_train.todense()
    x_test_dense = x_test.todense()
    x_val_dense = x_val.todense()

    x_train_tensor = torch.tensor(x_train_dense).float()
    x_test_tensor = torch.tensor(x_test_dense).float()
    x_val_tensor = torch.tensor(x_val_dense).float()

    input_size = x_train_dense.shape[1]
    print("Input size:", input_size)

    return x_train_tensor, x_test_tensor, x_val_tensor, input_size


def gen_tensor_from_arr(df, text_pipeline):
    text_list = []
    for func in df:
        processed_text = torch.tensor(text_pipeline(func)).float()
        text_list.append(processed_text)

    return torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in text_list], batch_first=True)


def get_features_tokenizer(x_train, x_test, x_val):
    tokenizer = get_tokenizer(tokenizer=None)

    def yield_tokens(data):
        for func in data:
            yield tokenizer(func)

    vocab = build_vocab_from_iterator(yield_tokens(np.concatenate((x_train, x_test, x_val))))
    text_pipeline = lambda x: vocab(tokenizer(x))

    # padding since we need all values to be same size
    all_tensor = gen_tensor_from_arr(np.concatenate((x_train, x_test, x_val)), text_pipeline)

    x_train_tensor = all_tensor[0:x_train.shape[0]]
    x_test_tensor = all_tensor[x_train.shape[0]:x_train.shape[0] + x_test.shape[0]]
    x_val_tensor = all_tensor[x_train.shape[0] + x_test.shape[0]:]

    input_size = x_train_tensor.shape[1]
    print("Input size:", input_size)
    return x_train_tensor, x_test_tensor, x_val_tensor, input_size


def get_data_loaders(
    x_train_tensor, x_test_tensor, x_val_tensor,
    y_train_tensor, y_test_tensor, y_val_tensor,
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    # Create train and validation datasets
    train_data = CodeDataset(x_train_tensor, y_train_tensor)
    valid_data = CodeDataset(x_val_tensor, y_val_tensor)

    train_idx = torch.randperm(len(train_data))
    valid_idx = torch.randperm(len(valid_data))

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    # drop_last is set to True to remove incomplete batches that may result in wrong input size
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        drop_last=True
    )

    # Now create the test data loader
    test_data = CodeDataset(x_test_tensor, y_test_tensor)

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True
    )

    return data_loaders

# def visualize_one_batch(data_loaders, max_n: int = 5):
#     """
#     Visualize one batch of data.
#
#     :param data_loaders: dictionary containing data loaders
#     :param max_n: maximum number of images to show
#     :return: None
#     """
#
#     # :
#     # obtain one batch of training images
#     # First obtain an iterator from the train dataloader
#     dataiter = iter(data_loaders["train"])
#     # Then call the .next() method on the iterator you just
#     # obtained
#     images, labels = dataiter.next()
#
#     # Undo the normalization (for visualization purposes)
#     mean, std = compute_mean_and_std()
#     invTrans = transforms.Compose(
#         [
#             transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
#             transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
#         ]
#     )
#
#     images = invTrans(images)
#
#     # :
#     # Get class names from the train data loader
#     class_names = data_loaders['train'].dataset.classes
#
#     # Convert from BGR (the format used by pytorch) to
#     # RGB (the format expected by matplotlib)
#     images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)
#
#     # plot the images in the batch, along with the corresponding labels
#     fig = plt.figure(figsize=(25, 4))
#     for idx in range(max_n):
#         ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
#         ax.imshow(images[idx])
#         # print out the correct label for each image
#         # .item() gets the value contained in a Tensor
#         ax.set_title(class_names[labels[idx].item()])
