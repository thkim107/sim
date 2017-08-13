import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class RNN_1(nn.Module):
    def __init__(self, input_size=12, hidden_size=2, num_layers=1, bidirectional=False):
        super(RNN_1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Set initial states
        if self.bidirectional == True:
            h0 = Variable(torch.randn(2 * self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.randn(2 * self.num_layers, x.size(0), self.hidden_size))
        else:
            h0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))
        
        # Forward propagate
        out, _ = self.rnn(x, (h0, c0))
        return out

class RNN_2(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, num_layers=1, bidirectional=False):
        super(RNN_2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)     # coversong & noncoversong

    def forward(self, x):
        # Set initial states
        if self.bidirectional == True:
            h0 = Variable(torch.randn(2 * self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.randn(2 * self.num_layers, x.size(0), self.hidden_size))
        else:
            h0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate
        out, _ = self.rnn(x, (h0, c0))

        # MLP with last step from RNN
        out = self.fc(out[:, -1, :])
        return out

# In the case of controlling inital input size to the minimum length of all inputs
# Directly apply a MLP network with the output of first RNN
class MLP(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_dims=[50, 10]):
        super(MLP, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out
