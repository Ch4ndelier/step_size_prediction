import torch
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size=1, num_layers=3):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            # dropout=0.5
        )
        self.linear = torch.nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out[-1])
        return linear_out, hidden
