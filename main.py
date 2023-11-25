import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--load', help='model to be loaded (random model is used if empty)')
parser.add_argument('-g', '--game-count', help='how many games to be played')
parser.add_argument('-s', '--save-every', help='how often to save the model')
parser.add_argument('-S', '--save-location', help='where to save the model')

args = parser.parse_args()

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

# save_path = "/Users/maximo/Documents/Class/aiml/tictactoe-ai/output/model"
print(args)
from training.gym import Gym
gym = Gym(NeuralNetwork,
    device=device,
    save_path=args.save_location,
    ep_count=args.game_count,
    save_every=args.save_every,
    load=args.load)
gym.train()
