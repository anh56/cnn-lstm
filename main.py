import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import get_data_loaders, get_data, get_features_tf_idf, get_features_tokenizer
from src.es import EarlyStopper
from src.helpers import get_data_location, save_metrics
from src.train import optimize, test
from src.optimization import get_optimizer, get_loss
from src.model import get_model


def gen_tok_pattern():
    single_toks = ['<=', '>=', '<', '>', '\\?', '\\/=', '\\+=', '\\-=', '\\+\\+', '--', '\\*=', '\\+', '-', '\\*',
                   '\\/', '!=', '==', '=', '!', '&=', '&', '\\%', '\\|\\|', '\\|=', '\\|', '\\$', '\\:']

    single_toks = '(?:' + '|'.join(single_toks) + ')'

    word_toks = '(?:[a-zA-Z0-9]+)'

    return single_toks + '|' + word_toks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", type=str, default="msr-vul.csv")
    parser.add_argument("--cvss_col", type=str)
    parser.add_argument("--arch", choices=["ffnn", "cnn", "lstm"], default="ffnn")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--out_path", type=str, default="./output")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--early_stopping_metrics", choices=["f1", "mcc", None], default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--loss_fn", type=str, default="nll")
    parser.add_argument("--test_only", default=False, action='store_true')

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))

    batch_size = 64  # size of the minibatch for stochastic gradient descent (or Adam)
    valid_size = 0.2  # fraction of the training data to reserve for validation
    num_epochs = args.num_epochs  # number of epochs for training
    dropout = 0.001  # dropout for our model
    learning_rate = 0.05  # Learning rate for SGD (or Adam)
    opt = args.optimizer  # optimizer. 'sgd' or 'adam'
    weight_decay = 0.001  # regularization. Increase this to combat overfitting
    hidden_size = 64
    momentum = 0.5

    base_output = Path(args.out_path)

    early_stopper = None
    output = base_output / f"{args.cvss_col}_{args.arch}_{opt}_{args.loss_fn}_e{num_epochs}"
    if args.early_stopping_metrics:
        early_stopper = EarlyStopper(
            early_stopping_metrics=args.early_stopping_metrics,
            patience=args.early_stopping_patience
        )
        output = base_output / f"{args.cvss_col}_{args.arch}_{opt}_{args.loss_fn}_es{args.early_stopping_metrics}"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Run result will be saved to {output}")

    train_df, test_df, val_df = get_data(args=args)
    x_train = train_df.code.values
    x_val = val_df.code.values
    x_test = test_df.code.values
    print("len(train_df)", len(train_df), "len(test_df)", len(test_df), "len(val_df)", len(val_df))
    print("train counter", Counter(train_df.category).most_common())
    print("val counter", Counter(val_df.category).most_common())
    print("test counter", Counter(test_df.category).most_common())
    y_train_tensor = torch.tensor(train_df.category.values.astype(np.int32))
    y_test_tensor = torch.tensor(test_df.category.values.astype(np.int32))
    y_val_tensor = torch.tensor(val_df.category.values.astype(np.int32))

    vocab_size = 0
    if args.arch == "ffnn":
        x_train_tensor, x_test_tensor, x_val_tensor, input_size = get_features_tf_idf(x_train, x_test, x_val)
    else:
        x_train_tensor, x_test_tensor, x_val_tensor, input_size, vocab_size = get_features_tokenizer(
            x_train, x_test, x_val
        )

    data_loaders = get_data_loaders(
        x_train_tensor=x_train_tensor,
        x_test_tensor=x_test_tensor,
        x_val_tensor=x_val_tensor,
        y_train_tensor=y_train_tensor,
        y_test_tensor=y_test_tensor,
        y_val_tensor=y_val_tensor,
        batch_size=batch_size,
        valid_size=valid_size,
        num_workers=1
    )

    model = get_model(
        arch=args.arch,
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=args.num_classes,
        dropout=dropout,
        vocab_size=vocab_size,
        embedding_dim=100,
        batch_size=batch_size
    )

    if not args.test_only:
        optimizer = get_optimizer(
            model=model, optimizer=opt, learning_rate=learning_rate,
            momentum=momentum, weight_decay=weight_decay
        )
        loss = get_loss(args.loss_fn)
        optimize(
            data_loaders, model, optimizer, loss, n_epochs=num_epochs,
            model_save_path=output / "best_val_loss.pt", result_save_path=output / f"val.csv",
            early_stopper=early_stopper
        )

    # test
    # load the weights in 'checkpoints/best_val_loss.pt'
    model.load_state_dict(torch.load(output / "best_val_loss.pt"))

    # Run test
    accuracy, precision, recall, f1, mcc = test(data_loaders['test'], model)
    save_metrics(accuracy, precision, recall, f1, mcc, output / f"test.csv")


if __name__ == '__main__':
    main()
