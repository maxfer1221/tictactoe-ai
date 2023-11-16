from game import Game

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
            ##############################################
            ## TODO: Delete
            ## temporary output for demonstration purposes
            print("===============")
            game.board.print()
            ##############################################
            agents[player].insert(to_bit_arr(decoded["state"], 20))
            net_out = agents[player].feed_forward()
            decoded = self.decode(game.turn(net_out))
            polayer = game.turns % 2
        

        ##############################################
        ## TODO: Delete
        ## temporary output for demonstration purposes
        print("===============")
        game.board.print()
        ##############################################

        return decoded

        # tbd
        #if decoded["err"]:
        #    return self.punish(game.turns)
        #if decoded["results"]["o_win"]:
        #    return self.reward(agents[0])
        #if decoded["results"]["x_win"]:
        #    return self.reward(agents[1])
        #return self.tie(agents)

    def decode(self, game_output):
        out = {}
        CODE = game_output >> 28
        out["cont"]    = CODE == 0b0000
        out["results"] = {
            "tie"  :  CODE == 0b0001,
            "o_win":  CODE == 0b0010,
            "x_win":  CODE == 0b0011,
        }
        out["err"]     = CODE >= 0b0100
        # keep this as a number until we are sure we need the vector input
        out["state"]   = (game_output >> 8) & ((1 << 20) - 1)
        return out

def to_bit_arr(num, keep):
    return [(num >> (keep - i - 1)) % 2 for i in range(keep)]

# came up with a picker method that should stop any rematches, and should make it
# so that the agents always play an equal number of games going first and second.
# speed is probably questionable. we'll discuss it in person. don't feel like writing
# it here since i'd need a larger "training" module to place it in
# (game count should be even, 
#  population should be divisible by (game_cnt * 2)
import random
def default_picked(population, game_cnt=10):
    random.shuffle(population)
    #etc 

