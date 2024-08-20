import torch
import torch.nn as nn



class LSTM_RGB(nn.Module):

    def __init__(
        self,
        input_size = 1024,
        hidden_size = 512,
        n_layers = 2,
        dropout = 0.33,
        num_classes = 8
    ):
        super(LSTM_RGB, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x
    ):
        if x.dim() == 2:
            x = x.unsqueeze(2)

        x = x.permute(2, 0, 1)

        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        
        out, (hidden, cell) = self.lstm(x, (hidden_state, cell_state))
        # to be reviewed
        #out, _ = self.lstm(x) 

        out = self.dropout(out[:, -1, :])
        out = self.relu(out)
        out = self.fc(out)

        return out, {}