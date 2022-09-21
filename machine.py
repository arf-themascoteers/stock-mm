import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.hidden_dim = 32
        self.num_layers = 2

        self.lstm = nn.LSTM(2, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h0_1 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        c0_1 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0_1.detach(), c0_1.detach()))
        out = self.fc(out[:,-1,:])
        out = out.squeeze(1)
        return out