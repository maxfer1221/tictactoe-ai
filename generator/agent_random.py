import random

# playing agent that places randomly on the board
# does not place in occupied spots
# used to heurestically generate board states
class AgentRandom:
    def __init__(self):
        pass

    def predict(self, state):
        available = [0,1,2,3,4,5,6,7,8]
        for i in range(18):
            spot = i % 9
            if state[i] == 1:
                available = list(filter(lambda x: x != spot, available))
        return random.choice(available)
