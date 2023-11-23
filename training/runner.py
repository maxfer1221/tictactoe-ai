from common import *
from tictactoe.game import Game

class Runner:
    def __init__(self, agents):
        self.game = Game()
        # we assume the agents are in the following order
        # [ agent to play first, agent to play second ]
        self.agents = agents

    def run(self):
        agents = self.agents
        game = self.game

        game_output = game.request_state(inc=False)
        decoded = self.decode(game_output)

        player = game.turns % 2
        while(decoded["cont"]):
            net_out = agents[player].predict(board_to_arr(decoded["state"], swapped=player == 1))
            decoded = self.decode(game.turn(net_out))
            player = game.turns % 2

        return decoded

    def decode(self, game_output):
        out = {}

        # keep this as a number until we are sure we need the vector input
        out["state"] = (game_output >> 10) & ((1 << 18) - 1)
        #

        CODE = game_output >> 28
        out["cont"] = CODE == 0b0000
        if out["cont"]:
            return out

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
        # print(out["turn_count"])

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
    # print(arr)
    return arr
