import torch
from torch import nn

class LSTM_EMG(nn.Module):
    def __init__(self, input_size = 16, hidden_size1 = 5, hidden_size2 = 50,
                  num_classes = 20, n_layers = 1, dropout = 0.2): #num_classes = 20 or 21?
        super(LSTM_EMG, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.lstm1 = nn.LSTM(input_size, self.hidden_size1, n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, self.hidden_size2, n_layers, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(self.hidden_size2, num_classes) 

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        out = self.dropout(out[:, -1, :])
        out = self.relu(out)

        out = self.fc(out)

        return out, {}