from common import *

import torch
from torch import nn
from training.runner import Runner
from training.agent import Agent
from functools import reduce
import random
from random import shuffle

# genetic algorithm utilities
import pygad
import pygad.torchga

class Gym:
    def __init__(self, NeuralNetwork, **kwargs):
        # declare seed for RNGs
        seed = kwargs.get('seed', torch.rand(1))
        torch.manual_seed(seed)

        # neural network to be used by gym
        self.NN = NeuralNetwork
        template = self.NN().named_parameters()
        # network parameter keys and sizes
        # parameter_sizes[key] = size
        self.parameter_sizes = {n:p.size() for n,p in template}

        self.device = kwargs.get('device', None)
        if self.device is None:
            raise UndefinedDeviceException

        # load population from folder
        load_from = kwargs.get('load_from', None)
        if load_from is not None:
            self.population = load_agents(NeuralNetwork, self.device, load_from)
            self.popsize = len(self.population)
        else:
            self.popsize = kwargs.get("popsize", 100)
            self.population = [Agent(NeuralNetwork, self.device) for _ in range(self.popsize)]

        self.game_count = kwargs.get("game_count", 20)
        self.save_path  = kwargs.get("save_path", None)

        kg = kwargs.get

        initial_population = pygad.torchga.TorchGA(model=NeuralNetwork().linear_relu_stack,
            num_solutions=self.popsize).population_weights

        fitness = lambda _,__,solution_idx: self.fitness_func(solution_idx)
        on_gen  = lambda g: on_generation(self, g)
        # for explanations of each parameter see
        # https://pygad.readthedocs.io/en/latest/pygad.html
        self.ga = pygad.GA(num_generations=kg("num_generations", 10000),
            initial_population=initial_population,
            num_parents_mating=kg("num_parents_mating", 5),
            fitness_func=kg("fitness_function", fitness),
            sol_per_pop=kg("sol_per_pop", self.popsize),
            num_genes=kg("num_genes", 9), # change this None once we know what it does
            init_range_low=kg("init_range_low", -4.0),
            init_range_high=kg("init_range_high", 4.0),
            parent_selection_type=kg("parent_selection_type", "sss"),
            keep_parents=kg("keep_parents", -1),
            keep_elitism=kg("keep_elitism", self.popsize // 20),
            crossover_type=kg("crossover_type", "single_point"),
            mutation_type=kg("mutation_type", "random"),
            mutation_percent_genes=kg("mutation_percent_genes", 10),
            on_generation=on_gen)

        self.generation = 0

    def train(self):
        self.ga.run()
        self.ga.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    def run_games(self):
        l = self.popsize//2
        # split generation in 2, face against each other
        results = []
        A, B = [], []
        for g in range(self.game_count):
            A, B = B, A
            if g % l == 0:
                shuffle(self.population)
                A = self.population[:l]
                B = self.population[l:]

            for i, agent in enumerate(A):
                adversary = B[(i + g) % l]
                r = Runner([agent, adversary])
                result = r.run()
                results.append(result)
                # result["board"].print()
                self.score(result, agent, adversary)

        min_fit = min(self.population, key=lambda x: x.fitness).fitness
        max_fit = max(self.population, key=lambda x: x.fitness).fitness
        for a in self.population:
            if max_fit > 0:
                a.fitness = max(a.fitness,0)
            else:
                a.fitness += abs(min_fit)

        avg_game_length = sum([result["turn_count"] for result in results])/len(results)
        max_game_length = max(results, key=lambda x: x["turn_count"])["turn_count"]
        print(f"average game length: {avg_game_length}, max game length: {max_game_length}")

    def score(self, result, a, b):
        if result["err"]:
            if result["err_type"] == OccupiedSpaceException:
                # [a,b][(result["last_played"] + 1) % 2].fitness -= 40 / (result["turn_count"]) ** 2
                [a,b][(result["last_played"] + 1) % 2].failed = True
                result["board"].print()
                print(result["last_played"])

        elif result["results"]["tie"]:
            a.fitness += 50
            b.fitness += 100

        elif result["results"]["o_win"]:
            a.fitness += 100

        elif result["results"]["x_win"]:
            a.fitness += 150

    def fitness_func(self, solution_idx):
        if self.population[solution_idx].failed:
            return 0
        return self.population[solution_idx].fitness

def save_models(models, path):
    [torch.save(model.state_dict(), f"{path}{i}")
        for i, model in enumerate(models)]

import os
def load_agents(NN, device, load_from):
    agents = []
    for filename in os.listdir(load_from):
        f = os.path.join(load_from, filename)
        # checking if it is a file
        if os.path.isfile(f):
            a = Agent(NN, device)
            a.model.load_state_dict(torch.load(f))
            agents.append(a)

    print("loaded population from files")
    return agents

from collections import OrderedDict
import numpy
def on_generation(gym, ga_instance):
    agents = []
    # weights is a list of floats holding the tensor entries for our network
    for weights in ga_instance.population:
        model_dict = OrderedDict()
        s_idx = 0
        for n,s in gym.parameter_sizes.items():
            if len(s) > 1:
                model_dict[n] = []
                for r in range(s[0]):
                    model_dict[n].append(weights[s_idx:s_idx + s[1]])
                    s_idx += s[0]
            else:
                model_dict[n] = weights[s_idx:s_idx + s[0]]
                s_idx += s[0]
            model_dict[n] = torch.tensor(numpy.array(model_dict[n]))
        agent = Agent(gym.NN, gym.device)
        agent.model.load_state_dict(model_dict)
        agents.append(agent)
    gym.population = agents

    gym.generation += 1
    print(f"running generation {gym.generation}...")
    gym.run_games() # run games and calculate fitness
    gym.population.sort(key=lambda x: x.fitness, reverse=True)
    print([a.fitness for a in gym.population[:5]])
    print([a.fitness for a in gym.population[len(gym.population)-5:]])

    if gym.save_path is not None and gym.generation % 10 == 0:
        save_models([p.model for p in gym.population], gym.save_path)
        print("models saved")
