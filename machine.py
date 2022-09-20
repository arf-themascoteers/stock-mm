import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.hidden_dim = 16
        self.num_layers = 1

        self.lstm1 = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 1)

        self.lstm2 = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.fc = nn.Linear(2,1)


    def forward(self, x):

        close = x[:,0]
        close = close.reshape(close.shape[0],close.shape[1], 1)

        h0_1 = torch.zeros(self.num_layers, close.shape[0], self.hidden_dim).requires_grad_()
        c0_1 = torch.zeros(self.num_layers, close.shape[0], self.hidden_dim).requires_grad_()
        out1, (hn, cn) = self.lstm1(close, (h0_1.detach(), c0_1.detach()))
        out1 = self.fc1(out1[:, -1, :])

        high = x[:,1]
        high = high.reshape(high.shape[0],high.shape[1], 1)

        h0_2 = torch.zeros(self.num_layers, high.shape[0], self.hidden_dim).requires_grad_()
        c0_2 = torch.zeros(self.num_layers, high.shape[0], self.hidden_dim).requires_grad_()
        out2, (hn, cn) = self.lstm2(high, (h0_2.detach(), c0_2.detach()))
        out2 = self.fc2(out2[:, -1, :])

        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)

        return out