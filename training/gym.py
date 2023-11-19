from common import *

import torch
from torch import nn
from training.runner import Runner
from training.agent import Agent
from functools import reduce
import random
from random import shuffle

class Gym:
    def __init__(self, NeuralNetwork, device, popsize=1000, **kwargs):
        seed = kwargs.get('seed', None)
        torch.manual_seed(seed if seed is not None else torch.rand(1))
        self.device = device
        self.NN = NeuralNetwork
        load_from = kwargs.get('load_from', None)
        if load_from is not None:
            self.population = load_agents(NeuralNetwork, device, load_from)
            self.popsize = len(self.population)
            print("loaded population from files")
        else:
            self.population = [Agent(NeuralNetwork, device) for _ in range(popsize)]
            self.popsize = popsize
        self.games_to_play = 200
        self.generation_count = 100000


        
    def train(self, **kwargs):
        for g in range(self.generation_count):
            print(f"running generation {g+1}...")
            self.run_and_score_generation()    # scoring
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            print([a.fitness for a in self.population[:5]])
            print([a.fitness for a in self.population[len(self.population)-5:]])
            self.population = self.breed(self.population) # crossover & mutation
            
            if g % 10 == 0:
                for i, agent in enumerate(self.population):
                    self.save_model(agent.model, i, path=kwargs.get('save_path', None))
                print("models saved")

    def run_and_score_generation(self):
        l = self.popsize//2
        # split generation in 2, face against each other
        results = []
        A, B = [], []
        for g in range(self.games_to_play):
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
                self.score(result, agent, adversary)

        avg_game_length = sum([result["turn_count"] for result in results])/len(results)
        max_game_length = max(results, key=lambda x: x["turn_count"])["turn_count"]
        print(f"average game length: {avg_game_length}, max game length: {max_game_length}")


    def score(self, result, a, b):
        if result["err"]:
            if result["err_type"] == OccupiedSpaceException:
                [a,b][result["last_played"]].fitness -= 40 / (result["turn_count"]) ** 2

        elif result["results"]["tie"]:
            a.fitness += 50
            b.fitness += 100

        elif result["results"]["o_win"]:
            a.fitness += 100

        elif result["results"]["x_win"]:
            a.fitness += 150

    def breed(self, population, mutate=True):
        new_pop = []

        weights = [a.fitness for a in population]
        m = min(weights)
        if max(weights) > 0:
            weights = [max(0, w) for w in weights]
        else:
            weights = [(w + abs(m)) ** 2 for w in weights]

        dicts = [a.model.state_dict() for a in population]

        for _ in range(len(population)):
            agent = Agent(self.NN, self.device)
            new_state_dict = OrderedDict()
            for k, _ in population[0].model.named_parameters():
                tensor = crossover([s[k] for s in dicts], weights)
                new_state_dict[k] = torch.tensor(tensor)
            agent.model.load_state_dict(new_state_dict)
            new_pop.append(agent)

        return new_pop

    def save_model(self, model, index, path=None):
        if path is not None:
            torch.save(self.population[index].model.state_dict(), f"{path}{index}")


from collections import OrderedDict
from numpy.random import normal
def crossover(arr, weights):
    if len(arr[0].shape) > 1:
        return [crossover([arr[i][j] for i in range(len(arr))], weights) for j in range(arr[0].shape[0])] 
    
    mutation_chance = 0.05
    return [random.choices([arr[i][j] for i in range(len(arr))], weights=weights)[0] 
        + normal() for j in range(arr[0].shape[0])
    ]

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
    return agents    
    