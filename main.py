# example runner usage
import random
from neural.network import NeuralNetwork

import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

from training.gym import Gym
gym = Gym(NeuralNetwork, device, popsize=80, seed=1)
gym.train(save_path="/home/maximo/Documents/class/aiml/project/tictactoe/new_output/parent_")