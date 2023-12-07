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

# used for multithreading
import multiprocessing as mp

# Gym class:
# training module for the neural network. handles how the neural network trains
class Gym:
    def __init__(self, NeuralNetwork, **kwargs):
        def kg(x, y):
            t = kwargs.get(x)
            return y if t is None else t

        # declare seed for RNGs
        seed = kg('seed', torch.rand(1))
        torch.manual_seed(seed)

        # neural network module to be used by gym
        self.NN = NeuralNetwork
        self.device = kg('device', None)
        if self.device is None:
            raise UndefinedDeviceException

        # generation population information
        self.popsize    = kg("popsize", 100)    # ignored if "load_from" is not None
        self.game_count = int(kg("game_count", 200))
        self.thread_count = min(kg("thread_count", mp.cpu_count()), mp.cpu_count())

        # load population from folder
        load_from = kg('load_from', None)
        if load_from is not None:
            ga = pygad.load(load_from)
            initial_population = ga.population
            self.popsize = len(ga.population)
        else:
            initial_population = torchga.TorchGA(self.NN(), num_solutions=self.popsize).population_weights

        # save information
        self.save_path = kg("save_path", "output")
        self.save_every = kg("save_every", 10) # number of generations to wait till saving

        # turn fitness and on-generation functions into lambdas for compliance with PyGAD structure
        fitness = lambda ga_instance,solution,solution_idx: self.fitness_func(ga_instance,solution,solution_idx)
        on_gen  = lambda g: on_generation(self, g)
        # for explanations of each parameter see
        # https://pygad.readthedocs.io/en/latest/pygad.html
        self.ga = pygad.GA(num_generations=kg("num_generations", 10000),
            initial_population=initial_population,
            num_parents_mating=kg("num_parents_mating", 5),
            fitness_func=kg("fitness_function", fitness),
            sol_per_pop=kg("sol_per_pop", self.popsize),
            parent_selection_type=kg("parent_selection_type", "tournament"),
            keep_parents=kg("keep_parents", -1),
            keep_elitism=kg("keep_elitism", self.popsize // 20),
            crossover_type=kg("crossover_type", "uniform"),
            mutation_type=kg("mutation_type", "random"),
            mutation_percent_genes=kg("mutation_percent_genes", 10),
            on_generation=on_gen,
            parallel_processing=['thread', self.thread_count])

        # generation counter
        self.generation = 0

        # information arrays
        self.results   = []
        self.fitnesses = []

        # whether to use predetermined board states for training
        game_states = kg("load_games", None)
        if game_states is not None:
            # used if specific games need to be played
            import ast
            with open(game_states,'r') as f:
                self.game_db = ast.literal_eval(f.read())

            game_index = kg("load_game_index", "0")
            self.games = self.game_db[game_index]["games"]
        else:
            self.games = [0] * self.game_count

    # function wrapper around the PyGAD training algorithm
    def train(self):
        print("running generation 0...")
        self.ga.run()

    def run_games(self, agent):
        for i in range(self.game_count):
            # generate a game "runner" that handles game IO with the network
            # the runner makes the agent play against an AI playing completely random moves
            r = Runner(agent, agent_first=True) # i%2 -> agent alternates between first and second
            result = r.run()

            if not result["err"]:
                agent.fitness += 1
                # agent goes first
                # if i%2 == 0:
                #     # winning as first is rewarded
                #     if result["results"]["o_win"]:
                #         agent.fitness += 1
                #     # tying as second is OK
                #     elif result["results"]["tie"]:
                #         agent.fitness += 1

                # agent goes second
                # else:
                #     # winning as second is highly rewarded
                #     if result["results"]["x_win"]:
                #         agent.fitness += 6
                #     # tying as second is also rewarded
                #     elif result["results"]["tie"]:
                #         agent.fitness += 3

            # general fitness calculation
            # if not result["err"]:
            #     agent.fitness += 1
            #     # agent goes first
            #     if i%2 == 0:
            #         # winning as first is rewarded
            #         if result["results"]["o_win"]:
            #             agent.fitness += 3
            #         # tying as second is OK
            #         elif result["results"]["tie"]:
            #             agent.fitness += 1

            #     # agent goes second
            #     else:
            #         # winning as second is highly rewarded
            #         if result["results"]["x_win"]:
            #             agent.fitness += 6
            #         # tying as second is also rewarded
            #         elif result["results"]["tie"]:
            #             agent.fitness += 3

            # result array used for output printing
            # self.results.append(result)
        # normalize the fitness
        agent.fitness /= len(self.games)

    # fitness function to be used by the PyGAD genetic algorithm
    def fitness_func(self, ga_instance, solution, solution_idx):
        # convert the PyGAD solution types into pytorch models
        nn = self.NN().to(self.device)
        model_dict = torchga.model_weights_as_dict(model=nn.model, weights_vector=solution)
        nn.model.load_state_dict(model_dict)

        # agent object that stores fitness. used by the game runner
        agent = Agent(self.NN, self.device, model=nn)
        self.run_games(agent)

        # print(agent.fitness)
        # add fitness to array (used for intermediate printing)
        self.fitnesses.append(agent.fitness)

        return agent.fitness

import numpy
import pygad.torchga as torchga
def on_generation(gym, ga_instance):
    if gym.generation > 0:
        avg_fitness = numpy.mean(gym.fitnesses)
        max_fitness = max(gym.fitnesses)
        print(f"average fitness: {avg_fitness}, max fitness: {max_fitness}")

    if gym.generation % gym.save_every:
        gym.ga.save(gym.save_path)
        print("models saved")

    gym.fitnesses = []
    gym.results = []
    gym.generation += 1
    print(f"running generation {gym.generation}...")
