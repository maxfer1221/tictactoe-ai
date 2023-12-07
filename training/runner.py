from common import *
from tictactoe.game import Game
from training.agent_random import AgentRandom

# Runner class
# runs the tic tac toe matches. handles IO from the game
class Runner:
    def __init__(self, agent, from_state=None, agent_first=True):
        self.game  = Game()
        self.agent = agent
        self.ad    = AgentRandom()
        self.agent_first = agent_first
        if from_state != None:
            self.game.set_state(from_state)

    # runs the tic tac toe game
    def run(self):
        agent = self.agent
        game = self.game

        game_output = game.request_state(inc=False)
        decoded = self.decode(game_output)

        agent_turn = self.agent_first

        # while the game is still going
        while(decoded["cont"]):
            if agent_turn:
                # give game the network's chosen move
                net_out = agent.predict(board_to_arr(decoded["state"], swapped=self.agent_first))
                decoded = self.decode(game.turn(net_out))
            else:
                # give game a random (valid) move to play
                move = self.ad.predict(board_to_arr(decoded["state"]))
                decoded = self.decode(game.turn(move))
            agent_turn = not agent_turn

        return decoded

    # decodes the game output
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

        out["turn_count"] = ((game_output >> 4) & (0b1111))

        player = (game_output >> 8) % 0b11
        out["last_played"] = 1 if player == 0b10 else 0

        out["board"] = self.game.board

        return out

# helper functions to interpret the game output
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
