import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class SingleLayeredNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_dim: int):
        super(SingleLayeredNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_dim)
        self.sigmoid2 = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        sigmoid = self.sigmoid1(hidden)
        output = self.fc2(sigmoid)
        output = self.sigmoid2(output)
        return output
