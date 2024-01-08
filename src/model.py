import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_model(args, input_size, hidden_size, dropout, vocab_size, embedding_dim, batch_size):
    model = None
    print(f"Building [{args.arch}] model")
    if args.multitask:
        match args.arch:
            case "ffnn":
                raise NotImplementedError()  # we dont implement ffnn multitask
            case "cnn":
                model = CNN_multitask(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    batch_size=batch_size
                )
            case "lstm":
                model = LSTM_multitask(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=2,
                    batch_size=batch_size
                )
            case "gru":
                model = GRU_multitask(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=2,
                    batch_size=batch_size
                )
            case _:
                return
    else:
        match args.arch:
            case "ffnn":
                model = FFNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_classes=args.num_classes,
                    dropout=dropout
                )
            case "cnn":
                model = CNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_classes=args.num_classes,
                    dropout=dropout,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    batch_size=batch_size
                )
            case "lstm":
                model = LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_classes=args.num_classes,
                    dropout=dropout,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=3,
                    batch_size=batch_size
                )
            case "gru":
                model = GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_classes=args.num_classes,
                    dropout=dropout,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_layers=2,
                    batch_size=batch_size
                )
    print(model)
    return model


class FFNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes: int = 3, dropout: float = 0.1
    ):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


class CNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes, dropout,
        vocab_size, embedding_dim, batch_size
    ):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.stride = 1
        self.padding = 0
        self.kernel_size = 3
        self.dilation = 1
        self.final_linear_input_size = int(
            (
                (
                    input_size + 2 * self.padding - self.dilation * (
                    self.kernel_size - 1) - 1
                ) / self.stride
            ) + 1
        ) * batch_size * 2  # conv1d output * batch size * number of dimension in conv1d (2)
        print("final_linear_input_size", self.final_linear_input_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.final_linear_input_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print("\n\n", x.shape)
        x = self.embedding(x).float()
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        # print(x.shape)
        # x = self.pool(x)
        x = self.dropout(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # flatten layer
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = self.log_softmax(x)
        # print(x.shape)
        return x


class MulticlassClassificationHead(nn.Module):
    def __init__(
        self, hidden_size,
        num_labels_per_category=[3, 3, 2, 3, 3, 3, 3]
    ):
        super().__init__()
        self.out_prj = nn.ModuleList(
            [
                nn.Linear(hidden_size, num_labels)
                for num_labels in num_labels_per_category
            ]
        )

    def forward(self, x):
        logits = [layer(x) for layer in self.out_prj]
        return logits


class CNN_multitask(nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout,
        vocab_size, embedding_dim, batch_size
    ):
        super(CNN_multitask, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=5, padding=2)
        # self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(388768, hidden_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.classifier = MulticlassClassificationHead(hidden_size)

    def forward(self, x):
        # print("x org", x.shape)
        x = self.embedding(x).float()
        # print("x embed", x.shape)
        x = x.permute(0, 2, 1)
        # print("x permute", x.shape)
        x = self.conv1(x)
        # print("x conv", x.shape)
        x = F.relu(x)
        # print("x relu", x.shape)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten layer
        # print("x view", x.shape)
        x = self.fc(x)
        # print("x fc", x.shape)
        x = self.log_softmax(x)
        # print("x log softmax", x.shape)
        x = self.classifier(x)
        # print("len(x)", len(x))
        return x


# class LSTM(nn.Module):
#     def __init__(
#         self, input_size, hidden_size, num_classes, dropout,
#         vocab_size, embedding_dim, num_layers, batch_size
#     ):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.num_classes = num_classes
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.fc = nn.Linear(hidden_size * 2, num_classes)
#         # self.log_softmax = nn.LogSoftmax(dim=1)
#
#         # self.relu = nn.ReLU()
#         if num_classes > 2:
#             self.fc = nn.Linear(hidden_size * 2, num_classes)
#             # self.fn = nn.LogSoftmax(dim=1)
#         else:
#             self.fc = nn.Linear(hidden_size * 2, 1)
#             self.fn = nn.Sigmoid()
#
#     # def forward(self, x):
#     #     x = self.embedding(x).float()
#     #     out, _ = self.lstm(x)
#     #     out = self.fc(out[:, -1, :])
#     #     # out = self.log_softmax(out)
#     #     return out
#
#     def forward(self, x):
#         x = self.embedding(x).float()
#         out, _ = self.lstm(x)
#         # print(out.shape)
#         out = out[:, -1, :]
#         # print(out.shape)
#         out = self.relu(out)
#         print(out)
#         print(out.shape)
#
#         out = self.fc(out)
#         print(out)
#         print(out.shape)
#
#         out = self.fn(out)
#         print(out)
#         print(out.shape)
#
#         # print(out)
#         # match the output
#         if self.num_classes > 2:
#             print(out)
#             print(out.shape)
#             return out
#         else:
#             out = out.squeeze(1)
#             return out

class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes, dropout,
        vocab_size, embedding_dim, num_layers, batch_size
    ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(hidden_size, num_classes)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x).float()
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        # out = out.view(out.size(0), -1)
        # out = self.log_softmax(out)
        return out



class LSTM_multitask(nn.Module):
    def __init__(
        self, input_size, hidden_size, dropout,
        vocab_size, embedding_dim, num_layers, batch_size
    ):
        super(LSTM_multitask, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(hidden_size, num_classes)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.classifier = MulticlassClassificationHead(hidden_size)

    def forward(self, x):
        x = self.embedding(x).float()
        # print(x.shape)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).cuda()
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        # print(out.shape)
        # out = self.fc(out)
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        # out = out.view(out.size(0), -1)
        # out = self.log_softmax(out)
        out = self.classifier(out)
        return out


class GRU(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes, dropout,
        vocab_size, embedding_dim, num_layers, batch_size
    ):
        super(GRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.embedding(x).float()
        # print(x.shape)
        x, _ = self.gru(x)
        # print(x.shape)
        # x = self.dropout(x)
        # print(x.shape)
        # print(x)
        # print(x[:, -1, :])
        x = self.fc(x[:, -1])
        # print(x.shape)
        # x = self.log_softmax(x)
        return x


class GRU_multitask(nn.Module):
    pass
