import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class SingleLayeredNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayeredNN, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = torch.nn.Linear(input_dim, output_dim)
        # Output layer, 10 units - one for each digit
        self.output = torch.nn.Linear(input_dim, output_dim)

        # Define sigmoid activation and softmax output
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x
