import torch
from torch import nn
import torch.nn.functional as F


class GRU_RGB(nn.Module):
    # GRU for RGB
    def __init__(self,  n_layers, dropout, input_dim=1024, hidden_dim=512, output_dim=8):
        super(GRU_RGB, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.hidden = self.init_hidden(32)
        
    def forward(self, x):
        self.hidden = torch.zeros((self.n_layers, x.size(0), self.hidden_dim)).to(self.device)
        sample = x.unsqueeze(1)
        out, self.hidden = self.gru(sample, self.hidden)
        out = self.fc(self.relu(out[:,-1]))
        return out, {"features": sample}
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
