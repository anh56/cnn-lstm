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

from src.data import get_data_loaders
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
    output = base_output / f"{args.cvss_col}_{args.arch}_{opt}_e{num_epochs}"
    if args.early_stopping_metrics:
        early_stopper = EarlyStopper(
            args.early_stopping_metrics,
            args.early_stopping_patience
        )
        output = base_output / f"{args.cvss_col}_{args.arch}_{opt}_es{args.early_stopping_metrics}"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Run result will be saved to {output}")

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

    x_train = train_df.code.values
    x_val = val_df.code.values
    x_test = test_df.code.values
    print("len(train_df)", len(train_df), "len(test_df)", len(test_df), "len(val_df)", len(val_df))
    print("train counter", Counter(train_df.category).most_common())
    print("val counter", Counter(val_df.category).most_common())
    print("test counter", Counter(test_df.category).most_common())

    # code_token_pattern = gen_tok_pattern()
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_val = vectorizer.transform(x_val)
    # max_features_row = np.argmax(np.sum(x_train, axis=1))
    print("len(vectorizer.vocabulary_)", len(vectorizer.vocabulary_))

    x_train_dense = x_train.todense()
    x_test_dense = x_test.todense()
    x_val_dense = x_val.todense()

    x_train_tensor = torch.tensor(x_train_dense).float()
    x_test_tensor = torch.tensor(x_test_dense).float()
    x_val_tensor = torch.tensor(x_val_dense).float()
    y_train_tensor = torch.tensor(train_df.category.values.astype(np.int32))
    y_test_tensor = torch.tensor(test_df.category.values.astype(np.int32))
    y_val_tensor = torch.tensor(val_df.category.values.astype(np.int32))

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
        input_size=x_train_dense.shape[1],
        hidden_size=hidden_size,
        num_classes=args.num_classes,
        dropout=dropout
    )

    optimizer = get_optimizer(
        model=model,
        optimizer=opt,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    loss = get_loss()

    optimize(
        data_loaders,
        model,
        optimizer,
        loss,
        n_epochs=num_epochs,
        model_save_path=output / "best_val_loss.pt",
        result_save_path=output / f"val.csv",
        early_stopper=early_stopper
    )

    # test
    model = get_model(
        arch=args.arch,
        input_size=x_train_dense.shape[1],
        hidden_size=hidden_size,
        num_classes=args.num_classes,
        dropout=dropout
    )

    # load the weights in 'checkpoints/best_val_loss.pt'
    model.load_state_dict(torch.load(output / "best_val_loss.pt"))

    # Run test
    accuracy, precision, recall, f1, mcc = test(data_loaders['test'], model)
    save_metrics(accuracy, precision, recall, f1, mcc, output / f"test.csv")


if __name__ == '__main__':
    main()
