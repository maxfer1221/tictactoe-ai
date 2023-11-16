# example runner usage
import random

class Agent:
    def __init__(self):
        # ...
        pass

    def insert(self, _):
        # ...
        pass

    def feed_forward(self):
        return random.randint(0,8)

from runner import Runner
a1 = Agent()
a2 = Agent()
r = Runner([a1, a2])
r.run()
