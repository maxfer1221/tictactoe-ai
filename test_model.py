# model testing module
import argparse
parser = argparse.ArgumentParser(prog='ttrain')
parser.add_argument('-L', '--load', help='model load location', default="output")
parser.add_argument('-g', '--game-count', help='how many games to play', default=200)
parser.add_argument('-l', '--load-games', help='load games to play. if this is set, -i (--game-index) must be set too')
parser.add_argument('-i', '--load-game-index', help='game index to load')
parser.add_argument('-a', '--alternate', type=int, help='whether the ai should alternate going first: 0 for false, 1 for true', default=1)
parser.add_argument('-c', '--compare', type=int, help='whether to run a random agent to compare results', default=1)
args = parser.parse_args()

import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# import neural network model
from neural.network import NeuralNetwork
from training.agent import Agent
# generate agent
agent = Agent(NeuralNetwork, device)

import pygad
import pygad.torchga as torchga
# load best solution from population into agent
ga = pygad.load(args.load)
w, _, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
model_dict = torchga.model_weights_as_dict(model=agent.model.model, weights_vector=w)
agent.model.model.load_state_dict(model_dict)

from training.runner import Runner
from training.agent_random import AgentRandom

# load a game library if we should play from predetermined board positions
if args.load_games is not None:
    import ast
    game_db = {}
    with open(args.load_games,'r') as f:
        game_db = ast.literal_eval(f.read())
    game_db = game_db[args.load_game_index]["games"]
# otherwise create empty boards
else:
    game_db = [0] * args.game_count

game_count = len(game_db)

def alternate(i):
    if int(args.alternate) == 0:
        return True
    return i%2 == 0

err_cnt = 0
results = {
    "wins_f": 0,
    "wins_s": 0,
    "loss_f": 0,
    "loss_s": 0,
    "ties_f": 0,
    "ties_s": 0,
}
print("running tests")
for i, g in enumerate(game_db):
    runner = Runner(agent, agent_first=alternate(i), from_state=g)
    suffix = "f" if alternate(i) else "s"
    out = runner.run()
    if out["err"]:
        err_cnt += 1

    if out["results"]["tie"]:
        results[f"ties_{suffix}"] += 1
    elif out["results"]["x_win"]:
        if suffix == "s":
            results[f"wins_s"] += 1
        else:
            results[f"loss_f"] += 1
    elif out["results"]["o_win"]:
        if suffix == "f":
            results[f"wins_f"] += 1
        else:
            results[f"loss_s"] += 1
    elif out["err"]:
        results[f"loss_{suffix}"] += 1
print("num errors: ", err_cnt)
print(results)

if int(args.compare) == 0:
    print("num_games", game_count)
    exit()

# import a falliable agent to compare results
from training.agent_random_falliable import AgentRandomFalliable
err_cnt = 0
results = {
    "wins_f": 0,
    "wins_s": 0,
    "loss_f": 0,
    "loss_s": 0,
    "ties_f": 0,
    "ties_s": 0,
}
agent_random_falliable = AgentRandomFalliable()
for i, g in enumerate(game_db):
    runner = Runner(agent_random_falliable, agent_first=alternate(i), from_state=g)
    suffix = "f" if alternate(i) else "s"
    out = runner.run()
    if out["err"]:
        err_cnt += 1

    if out["results"]["tie"]:
        results[f"ties_{suffix}"] += 1
    elif out["results"]["x_win"]:
        if suffix == "s":
            results[f"wins_s"] += 1
        else:
            results[f"loss_f"] += 1
    elif out["results"]["o_win"]:
        if suffix == "f":
            results[f"wins_f"] += 1
        else:
            results[f"loss_s"] += 1
    elif out["err"]:
        results[f"loss_{suffix}"] += 1
print("num errors: ",  err_cnt)
print(results)
print("num games: ", game_count)
