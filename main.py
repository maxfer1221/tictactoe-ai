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

save_path = "/Users/maximo/Documents/Class/aiml/tictactoe-ai"

from training.gym import Gym
gym = Gym(NeuralNetwork,
    device=device,
    save_path=save_path,
    keep_elitism=1,
    parent_selection_type="rws",
    # crossover_type="uniform",
    popsize=100)
gym.train()
