import argparse
parser = argparse.ArgumentParser(prog='ttrain')
parser.add_argument('-s', '--seed', help='seed', default=1)
parser.add_argument('-L', '--load-from', help='model to be loaded (random model is used if empty)')
parser.add_argument('-g', '--game-count', help='how many games to be played', default=200)
parser.add_argument('--save-every', help='how often to save the model', default=10)
parser.add_argument('-S', '--save-path', help='where to save the model', default="output/pygad_output")
parser.add_argument('-N', '--num-generations', help='number of generations to train', default=1000)
parser.add_argument('-n', '--num-parents-mating', default=5)
parser.add_argument('-k', '--keep-elitism', help='how many of the best performers to keep', default=5)
parser.add_argument('-m', '--mutation-type', help='mutation method', default="random")
parser.add_argument('-c', '--crossover-type', help='crossover method', default="uniform")
parser.add_argument('-p', '--popsize', help='population size per generation. ignored if --load is set', default=100)
parser.add_argument('-l', '--load-games', help='load games to play. if this is set, -i (--game-index) must be set too')
parser.add_argument('-i', '--load-game-index', help='game index to load')
parser.add_argument('-t', '--thread-count', help='thread count. defaults to cpu virtual thread count', default=None)
args = parser.parse_args()

import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

from training.gym import Gym
from neural.network import NeuralNetwork
gym = Gym(NeuralNetwork, device=device, **vars(args))
gym.train()
