import torch

class Agent:
    def __init__(self, NeuralNetwork, device):
        self.fitness = 0
        self.device  = device
        self.model   = NeuralNetwork().to(device)

    def predict(self, state):
        X = torch.tensor([state], device=self.device)
        logits = self.model(X)
        pred_probab = self.model.Softmax(dim=1)(logits)
        return pred_probab.argmax(1)