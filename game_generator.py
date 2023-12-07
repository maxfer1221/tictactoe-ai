import argparse
parser = argparse.ArgumentParser(prog='ttrain')
parser.add_argument('-g', '--game-count', help='how many games to generate. if -1, will generate a (probabilistic) maximum amount. if set, may not converge and might run forever', default=-1)
parser.add_argument('-t', '--turn-count', help='required. how many moves should be played in the game', required=True)
parser.add_argument('-l', '--location', help='required. where to store/load games from. make a file with \'{}\' to start with a blank slate', required=True)
args = parser.parse_args()

# module to generate playable board states
from tictactoe.game import Game
from generator.agent_random import AgentRandom

# convert board to an array that can be interpreted by the network
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


# helper funtion to turn number to a bit array
def to_bit_arr(num, keep):
    arr = [1.0 * ((num >> (keep - i - 1)) % 2) for i in range(keep)]
    # print(arr)
    return arr

# convert game output into usable dict
def decode(game_output, game):
    out = {}

    # keep this as a number until we are sure we need the vector input
    out["state"] = (game_output >> 10) & ((1 << 18) - 1)
    #

    CODE = game_output >> 28
    out["cont"] = CODE == 0b0000

    out["results"] = {
        "tie"  :  CODE == 0b0001,
        "o_win":  CODE == 0b0010,
        "x_win":  CODE == 0b0011,
    }
    out["err"]     = CODE >= 0b0100
    if out["err"]:
        if CODE == 0b0100:
            out["err_type"] = ""
        else:
            out["err_type"] = "unknown"

    out["turn_count"] = ((game_output >> 5) & (0b111)) + 1
    # print(out["turn_count"])

    player = (game_output >> 8) % 0b11
    out["last_played"] = 1 if player == 0b10 else 0

    out["board"] = game.board

    # out["state"] = board_to_arr(out["state"])
    return out


game_count = args.game_count
turn_count = args.turn_count
games = {}
import ast
with open(args.location,'r') as f:
    games = ast.literal_eval(f.read())
games[f"{turn_count}"] = {
    "game_count": game_count,
    "games": set({})
}

# method to generate games until we can't find any new ones
# checks if game count has not changed in a long time
large_num = 10000
persistent = 0
iter = 0
def large_amnt(x):
    global persistent, large_num, iter
    if persistent == x:
        iter += 1
    else:
        persistent = x
        iter = 0

    if iter > large_num:
        return False
    return True

def lt(x, y):
    return x < y

conditional = lambda _: _
if game_count == -1:
    conditional = lambda x: large_amnt(x)
else:
    conditional = lambda x: lt(x, int(game_count))

print("generating games")

a = AgentRandom()
while conditional(len(games[f"{turn_count}"]["games"])):
    g = Game()
    d = decode(g.request_state(inc=False), g)
    for i in range(int(turn_count)):
        d = decode(g.turn(a.predict(board_to_arr(d["state"]))), g)
        if d["err"] or not d["cont"]:
            break

    if not d["err"] and d["cont"]:
        games[f"{turn_count}"]["games"].add(d["state"])

print(f'{len(games[f"{turn_count}"]["games"])} games generated')
with open('game_list.txt','w') as f:
   f.write(str(games))
