from common import *
from tictactoe.game import Game
from training.agent_random import AgentRandom
import torch
import numpy as np

from typing import Tuple
class Runner:
    def __init__(self, agent, agent_first=True):
        self.game  = Game()
        self.agent = agent
        self.adv   = AgentRandom()
        self.agent_first = agent_first
        self.turn = 0 if agent_first else 1

    def run_episode(self) -> Tuple[float, str]:
        err_cnt    = 0
        decoded    = self.decode(self.game.request_state(inc=False))
        next_state = []
        state      = decoded["state"]
        board_arr  = board_to_arr(state, swapped=not    self.agent_first)
        while decoded["cont"]:
            # network is playing
            if self.turn % 2 == 0:
                action_probs = self.agent.probs(board_arr).squeeze()
                log_probs = torch.log(action_probs)
                cpu_action_probs = action_probs.detach().cpu().numpy()

                action = np.random.choice(np.arange(9), p=cpu_action_probs)
                next_state = self.game.turn(action)
                decoded = self.decode(next_state)
                while decoded["err"]:
                    err_cnt += 1
                    action = np.random.choice(np.arange(9), p=cpu_action_probs)
                    next_state = self.game.turn(action)
                    decoded = self.decode(next_state)

            # random placement
            else:
                action = self.adv.predict(board_to_arr(decoded["state"], swapped=False))
                next_state = self.game.turn(action)

            decoded   = self.decode(next_state)
            state     = decoded["state"]
            board_arr = board_to_arr(state, swapped=self.agent_first == 1)

            self.turn += 1

        if decoded["err"]:
            return err_cnt, "err"
        if decoded["results"]["tie"]:
            return err_cnt, "tie"
        if decoded["results"]["x_win"]:
            return err_cnt, "win_scnd"
        if decoded["results"]["o_win"]:
            return err_cnt, "win_frst"
        return 0, " "

    def decode(self, game_output):
        out = {}
        out["err"] = False

        # keep this as a number until we are sure we need the vector input
        out["state"] = (game_output >> 10) & ((1 << 18) - 1)
        #

        CODE = game_output >> 28
        out["cont"] = CODE == 0b0000
        # if out["cont"]:
        #     return out

        out["results"] = {
            "tie"  :  CODE == 0b0001,
            "o_win":  CODE == 0b0010,
            "x_win":  CODE == 0b0011,
        }
        out["err"]     = CODE >= 0b0100
        if out["err"]:
            if CODE == 0b0100:
                out["err_type"] = OccupiedSpaceException
            else:
                out["err_type"] = "unknown"

        out["turn_count"] = ((game_output >> 5) & (0b111)) + 1

        player = (game_output >> 8) % 0b11
        out["last_played"] = 1 if player == 0b10 else 0

        out["board"] = self.game.board
        return out

def board_to_arr(bits, swapped=False):
    out_bits = 0
    for i in range(1,10):
        if (bits >> (18 - i * 2)) & 0b11 == 0b10:
            out_bits += 1
        out_bits <<= 1

    for i in range(1,10):
        if (bits >> (18 - i * 2)) & 0b11 == 0b01:
            out_bits += 1
        out_bits <<= 1


    arr = to_bit_arr(out_bits >> 1, 18)
    if swapped:
        arr = arr[9:] + arr[:9]

    return arr

def to_bit_arr(num, keep):
    arr = [1.0 * ((num >> (keep - i - 1)) % 2) for i in range(keep)]
    return arr
