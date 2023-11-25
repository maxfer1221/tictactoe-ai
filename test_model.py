import sys
model_loc = sys.argv[1]

import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

from neural.network import NeuralNetwork
from training.agent import Agent
agent = Agent(NeuralNetwork, device)
agent.model.load_state_dict(torch.load(model_loc))

print(f"loaded model from\n{model_loc}\nrunning simulations...")

from tests.test_runner import Runner
from training.agent_random import AgentRandom
game_count = 1000

err_cnt = 0
results = {
    "wins_f": 0,
    "wins_s": 0,
    "loss_f": 0,
    "loss_s": 0,
    "ties_f": 0,
    "ties_s": 0,
}
for i in range(game_count):
    runner = Runner(agent, agent_first=i % 2 == 0)
    suffix = "f" if i % 2 == 0 else "s"
    e, r = runner.run_episode()
    err_cnt += e
    if r == "tie":
        results[f"ties_{suffix}"] += 1
    elif r == "win_scnd":
        if suffix == "s":
            results[f"wins_s"] += 1
        else:
            results[f"loss_f"] += 1
    elif r == "win_frst":
        if suffix == "f":
            results[f"wins_f"] += 1
        else:
            results[f"loss_s"] += 1

print("average num errors: ", err_cnt / game_count)
print(results)
