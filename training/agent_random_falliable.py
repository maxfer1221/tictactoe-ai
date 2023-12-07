import torch

# agent that is able to make mistakes. used to test against the trained network
# for example, we can compre the number of errors made by an untrained random player to that of our network
import random
class AgentRandomFalliable:
    def __init__(self):
        pass

    def predict(self, state):
        available = [0,1,2,3,4,5,6,7,8]
        return random.choice(available)
