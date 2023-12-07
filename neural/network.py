from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(18, 15),  # 18 input neurons: 9 for for 'x' placement, 9 for 'o'
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 9),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)
