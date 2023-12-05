import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_model(arch, input_size, hidden_size, num_classes, dropout, vocab_size, embedding_dim, batch_size):
    model = None
    print(f"Building [{arch}] model")
    if arch == "ffnn":
        model = FFNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )
    elif arch == "cnn":
        model = CNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            batch_size=batch_size
        )
    else:
        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_size, kernel_size=5, padding=2)
        # self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(dropout)

        # # Calculate the output size after Conv1d
        # dummy_input = torch.zeros(1, input_size, dtype=torch.long)
        # dummy_output = self.conv1(self.embedding(dummy_input).float().permute(0, 2, 1))
        # output_size = dummy_output.view(1, -1).size(0)

        self.fc = nn.Linear(777536, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # self.conv1d_list = nn.ModuleList([
        #     nn.Conv1d(in_channels=self.embedding_dim,
        #               out_channels=num_filters[i],
        #               kernel_size=filter_sizes[i])
        #     for i in range(len(filter_sizes))
        # ])
        # # Fully-connected layer and Dropout
        # self.fc = nn.Linear(np.sum(num_filters), num_classes)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # print(x.shape)
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
        x = self.fc(x)
        # print(x.shape)
        x = self.log_softmax(x)
        return x

        # # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        # x_embed = self.embedding(x).float()
        #
        # # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # # Output shape: (b, embed_dim, max_len)
        # x_reshaped = x_embed.permute(0, 2, 1)
        #
        # # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        # x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        #
        # # Max pooling. Output shape: (b, num_filters[i], 1)
        # x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #                for x_conv in x_conv_list]
        #
        # # Concatenate x_pool_list to feed the fully connected layer.
        # # Output shape: (b, sum(num_filters))
        # x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
        #                  dim=1)
        #
        # # Compute logits. Output shape: (b, n_classes)
        # logits = self.fc(self.dropout(x_fc))
        #
        # return logits


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes: int = 3, dropout: float = 0.1
    ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        h0 = torch.zeros(self.num_layers, self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, self.hidden_dim).cuda()
        out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
        # out = out[-1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.log_softmax(out)
        return out
