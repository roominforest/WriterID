import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM, self).__init__()
        # self.bidirectional = bidirectional
        self.n_layer = n_layer
        # dimensions of the input feature
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        # self.out = nn.Linear(hidden_dim, n_class)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()  # consider bi
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()

        out, _ = self.lstm(x, (h0, c0))  # output of LSTM (batch_size, seq_length, hidden_size*2)

        out = self.fc(out[:, -1, :])  # the last hidden layer
        return out
