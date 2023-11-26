from common import *

import torch
from torch import nn
from training.runner import Runner
from training.agent import Agent
from training.agent_random import AgentRandom

import random
import numpy as np

from tqdm import tqdm
from threading import Thread

import subprocess
class Gym:
    def __init__(self, NeuralNetwork, **kwargs):
        # declare seed for RNGs
        seed = kwargs.get('seed', torch.rand(1))
        torch.manual_seed(seed)

        # neural network to be used by gym
        self.NN = NeuralNetwork

        self.device = kwargs.get('device', None)
        if self.device is None:
            raise UndefinedDeviceException

        # agent to train
        self.agent = Agent(self.NN, self.device)
        load_from  = kwargs.get("load", None)
        if load_from is not None:
            self.agent.model.load_state_dict(torch.load(load_from))

        # agent to play against
        self.adv = AgentRandom()

        self.ep_count = int(kwargs.get("ep_count", 10000))

        self.save_path  = kwargs.get("save_path" , None)
        self.save_every = int(kwargs.get("save_every", self.ep_count))
        missing_path    = (self.save_every is not None and self.save_path is None)
        missing_every   = (self.save_every is None and self.save_path is not None)
        if missing_path or missing_every:
            raise MissingPathorEveryException

        self.test = kwargs.get("test", False)

    def train(self):
        lengths = []
        rewards = []

        gamma = 0.99
        lr = 2**-13
        optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=lr)

        prefix = "reinforce-per-step"
        for ep_num in tqdm(range(1, self.ep_count + 1)):
            all_iterations = []
            all_log_probs  = []
            episode = list(self.generate_episode())
            raw_rewards = [reward for (_, __, reward), ___ in episode[:-1]]
            lengths.append(len(episode))
            for t, ((state, action, reward), log_probs) in enumerate(episode[:-1]):
                gammas_vec = gamma ** (torch.arange(t+1, len(episode))-t-1)
                # Since the reward is -1 for all st eps except the last, we can just sum the gammas
                G = -torch.sum(gammas_vec * torch.tensor(raw_rewards[t:]))
                rewards.append(G.item())
                policy_loss = log_probs[action]
                optimizer.zero_grad()
                gradients_wrt_params(self.agent.model, policy_loss)
                update_params(self.agent.model, lr  * G * gamma**t)
            self.generate_episode()
            if (ep_num % self.save_every == 0):
                self.save_model(ep_num, test=self.test)

    # runs 1 game, going first/second is random
    def generate_episode(self):
        agent_first = not random.getrandbits(1)
        r = Runner(self.agent, agent_first=agent_first)
        for v in r.run_episode():
            yield v

    def thread_test(self, i):
        process = subprocess.run(['python3', 'test_model.py', self.save_path+f"_{i}"], capture_output=True, text=True)
        print(process.stdout)

    def save_model(self, i, test=False):
        torch.save(self.agent.model.state_dict(), self.save_path+f"_{i}")
        if test:
            thread = Thread(target=self.thread_test, args=[i])
            thread.start()

from torch.autograd import grad
def gradients_wrt_params(net, loss_tensor):
    for name, param in net.named_parameters():
        g = grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = g

def update_params(net, lr):
    for name, param in net.named_parameters():
        param.data += lr * param.grad

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
