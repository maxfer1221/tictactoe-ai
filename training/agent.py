import torch

class Agent:
    def __init__(self, NeuralNetwork, device, **kwargs):
        self.fitness = 0.000001
        self.device  = device
        model = kwargs.get("model", None)
        if model is not None:
            self.model = model
        else:
            self.model = NeuralNetwork().to(device)

    def probs(self, state):
        X = torch.tensor([state], device=self.device)
        return self.model(X)
