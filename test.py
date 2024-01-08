# import argparse
# import json
# import os
# import pprint
# from collections import Counter
# from pathlib import Path
#
# import pandas as pd
# import sys
# import scipy
#
# import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from torch import nn, optim
# from tqdm import tqdm
#
# from data import get_data_location
#
# arch = ["cnn", "lstm"]
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cvss_col", type=str)
#     parser.add_argument("--arch", choices=arch, default="cnn")
#     parser.add_argument("--test_size", type=float, default=0.1)
#     parser.add_argument("--val_size", type=float, default=0.1)
#     parser.add_argument("--out_path", type=str, default="./output")
#
#     args = parser.parse_args()
#     output = Path(args.out_path)
#     output = output / f"{args.cvss_col}_{args.arch}"
#     output.mkdir(parents=True, exist_ok=True)
#
#     print(json.dumps(vars(args), indent=4))
#     data_folder = get_data_location()
#     df = pd.read_csv(data_folder + "/msr-vul.csv")
#     df = df[["func_before", args.cvss_col]].rename(
#         columns={
#             "func_before": "code",
#             args.cvss_col: "category"
#         }
#     )
#     print(df.shape, Counter(df["category"]).most_common())
#
#     train_df, tmp_df = train_test_split(
#         df,
#         test_size=args.test_size + args.val_size,
#         stratify=df.category,
#         random_state=42
#     )
#
#     val_df, test_df = train_test_split(
#         tmp_df,
#         test_size=args.val_size / (args.test_size + args.val_size),
#         stratify=tmp_df.category,
#         random_state=42
#     )
#     x_train = train_df.code.values
#     x_val = val_df.code.values
#     x_test = test_df.code.values
#     print("len(train_df)", len(train_df), "len(test_df)", len(test_df), "len(val_df)", len(val_df))
#     print("train counter", Counter(train_df.category).most_common())
#     print("val counter", Counter(val_df.category).most_common())
#     print("test counter", Counter(test_df.category).most_common())
#
#     vectorizer = TfidfVectorizer()
#     x_train = vectorizer.fit_transform(x_train)
#     x_test = vectorizer.fit_transform(x_test)
#     x_val = vectorizer.fit_transform(x_val)
#
#     x_train_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
#     x_test_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()
#     x_val_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(x_val)).float()
#
#     y_train_tensor = torch.tensor(train_df.category.values)
#     y_test_tensor = torch.tensor(test_df.category.values)
#     y_val_tensor = torch.tensor(val_df.category.values)
#
#     model = nn.Sequential(
#         nn.Linear(x_train_tensor.shape[1], 64),
#         nn.ReLU(),
#         nn.Dropout(0.1),
#         nn.Linear(64, df.category.nunique()),
#         nn.LogSoftmax(dim=1)
#     )
#
#     if torch.cuda.is_available():
#         model = model.cuda()
#
#     # Define the loss
#     criterion = nn.NLLLoss()
#
#     # Forward pass, get our logits
#     logps = model(x_train_tensor)
#     # Calculate the loss with the logits and the labels
#     loss = criterion(logps, y_train_tensor)
#
#     loss.backward()
#
#     # Optimizers require the parameters to optimize and a learning rate
#     optimizer = optim.Adam(model.parameters(), lr=0.002)
#
#     train_losses = []
#     val_losses = []
#     val_accuracies = []
#
#     epochs = 10
#     for e in tqdm(range(epochs)):
#         optimizer.zero_grad()
#
#         if torch.cuda.is_available():
#             x_train_tensor, y_train_tensor = x_train_tensor.cuda(), y_train_tensor.cuda()
#
#         output = model.forward(x_train_tensor)
#         loss = criterion(output, y_train_tensor)
#         loss.backward()
#         train_loss = loss.item()
#         train_losses.append(train_loss)
#
#         optimizer.step()
#
#         # Turn off gradients for validation, saves memory and computations
#         with torch.no_grad():
#             model.eval()
#
#             log_ps = model(x_val_tensor)
#             val_loss = criterion(log_ps, y_val_tensor)
#             val_losses.append(val_loss)
#
#             ps = torch.exp(log_ps)
#             top_p, top_class = ps.topk(1, dim=1)
#             equals = top_class == y_test_tensor.view(*top_class.shape)
#             val_accuracy = torch.mean(equals.float())
#             val_accuracies.append(val_accuracy)
#
#         model.train()
#
#         print(f"Epoch: {e + 1}/{epochs}.. ",
#               f"Training Loss: {train_loss:.3f}.. ",
#               f"Test Loss: {val_loss:.3f}.. ",
#               f"Test Accuracy: {val_accuracy:.3f}")
#
#
#     with torch.no_grad():
#         model.eval()
#         log_ps = model(x_val_tensor)
#         val_loss = criterion(log_ps, y_val_tensor)
#         val_losses.append(val_loss)
#
#         ps = torch.exp(log_ps)
#         top_p, top_class = ps.topk(1, dim=1)
#         equals = top_class == y_test_tensor.view(*top_class.shape)
#         val_accuracy = torch.mean(equals.float())
#         val_accuracies.append(val_accuracy)
#
#
#
# if __name__ == '__main__':
#     main()
