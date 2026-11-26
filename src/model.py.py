# RNN CELLimport torch
import torch.nn as nn
import math


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, h_prev):
        return torch.tanh(
            x @ self.W.T + h_prev @ self.U.T + self.b
        )


# LSTM CELL
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, h_prev, c_prev):
        gates = x @ self.W.T + h_prev @ self.U.T + self.b
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


# LSTM LAYER
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)

        outputs = []

        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


# BIDIRECTIONAL LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)

    def forward(self, x):
        forward_output = self.forward_lstm(x)

        reversed_x = torch.flip(x, dims=[1])
        backward_output = self.backward_lstm(reversed_x)
        backward_output = torch.flip(backward_output, dims=[1])

        return torch.cat(
            [forward_output, backward_output], dim=2
        )


# ATTENTION
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Parameter(torch.empty(hidden_size, 2 * hidden_size))
        self.ba = nn.Parameter(torch.empty(hidden_size))
        self.v = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(64)
        nn.init.uniform_(self.Wa, -bound, bound)
        nn.init.uniform_(self.ba, -bound, bound)
        nn.init.uniform_(self.v, -bound, bound)

    def forward(self, H):
        scores = torch.tanh(H @ self.Wa.T + self.ba) @ self.v
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha.unsqueeze(-1) * H, dim=1)
        return context, alpha


# COMPLETE HARNET
class HARNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_classes=6):
        super().__init__()
        self.bilstm = BiLSTM(input_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        H = self.bilstm(x)
        context, alpha = self.attention(H)
        logits = self.fc(context)
        return logits, alpha


