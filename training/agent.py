import torch

# Agent class
# wraps the neural network and adds a fitness
class Agent:
    def __init__(self, NeuralNetwork, device, **kwargs):
        self.fitness = 0.000001
        self.device  = device
        model = kwargs.get("model", None)
        if model is not None:
            self.model = model
        else:
            self.model = NeuralNetwork().to(device)
        self.failed = False

    # essentially passes
    def predict(self, state):
        X = torch.tensor([state], device=self.device)
        return self.model(X).argmax(1)
