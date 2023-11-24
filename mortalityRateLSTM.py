import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ExponentialActivation(nn.Module):
    def __init__(self):
        super(ExponentialActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)

# Define the LSTM model
class MortalityRateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, batch_first = True):
        super(MortalityRateLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first = batch_first)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first = batch_first)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first = batch_first)
        #self.concat = nn.Linear(hidden_size3 + 1, hidden_size3 + 1)
        self.output = nn.Linear(hidden_size3 + 1, 1)
        self.activation = ExponentialActivation()

        # Initialize weights
        y0=4.585904379419358 # Constant
        weights = np.zeros((1,hidden_size3 + 1))
        bias = np.log(y0)

        self.output.weight.data.copy_(torch.Tensor(weights))
        self.output.bias.data.copy_(torch.Tensor(np.array(bias)))


    def forward(self, x, gender):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = torch.cat((out[:, -1, :], gender.unsqueeze(1)), dim=1)
        #out = self.concat(out)
        out = self.output(out)
        out = self.activation(out)
        return out

if __name__ == '__main__':
    # Define hyperparameters
    input_size = 5
    hidden_size1 = 20
    hidden_size2 = 15
    hidden_size3 = 10

    # Create the model
    model = MortalityRateLSTM(input_size, hidden_size1, hidden_size2, hidden_size3)

    # Define the input tensors
    input1 = torch.randn(1, 10, 5)  # assuming batch size is 1
    gender = torch.randn(1)  # assuming batch size is 1

    # Forward pass
    output = model(input1, gender)

    print(output)