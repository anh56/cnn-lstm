import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_model(arch, input_size, hidden_size, num_classes, dropout):
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
            dropout=dropout
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
        self, input_size, hidden_size, num_classes: int = 3, dropout: float = 0.1
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        # self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.transpose(0, 1)  # swap channels and length, since tf-idf are in format of num_col, num_features
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)  # flatten layer
        x = self.dropout(x)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x


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
