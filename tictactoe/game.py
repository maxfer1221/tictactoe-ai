#########################################################
# Tic Tac Toe runner
# Simulates a Tic Tac Toe game. Converts the board and
# pertinent information into a single 32-bit integer,
# the outputs of which will be used by the game runner
# and neural network to train.
#
# See bottom for an explanation of this class's outputs
#########################################################
from common import * # common utils among the files
from tictactoe.board import Board

class Game:
    def __init__(self):
        # initialize board
        self.board = Board()
        
        # dictates which player is placing
        # % 2 = 0: 'o', % 2 = 1: 'x'
        # used to call a tie: 9 -> tie
        self.turns = 0

    # Primary function for the class
    # Parameters:
    #   - loc: int (0-8) Indicates position on 3x3 to play:  0 1 2
    # Returns:                                               3 4 5
    #   - 32-bit int: Interpretation explained at bottom     6 7 8
    #                 of file
    def turn(self, loc):
        try:
            self.board.put_checked(loc // 3, loc % 3, self.turns % 2)
        except OccupiedSpaceException as e:
            return self.err(OccupiedSpaceException)
        
        return self.request_state(inc=True)
    
    def request_state(self, inc=False):
        if self.board.win_on_board():
            return self.win()

        if self.turns == 8:
            return self.tie()

        if inc:
            self.turns += 1

        return self.expect_next()

    def win(self):
        # self.board.print()
        out = 0b0010
        if self.turns % 2 == 1:
            out += 0b0001

        # shift 18 bits to make space for game board
        out <<= 18
        # add board bits to output
        out += self.board_to_bits()
        # add "player turn" bits
        out <<= 2
        out += 0b01 if self.turns % 2 == 0 else 0b10
        # fix spacing (makes the leftmost 4 bits the output code)
        out <<= 8

        return out

    # see 'win' method for explanation
    def tie(self):
        # self.board.print()
        out = 0b0001
        out <<= 18
        out += self.board_to_bits()
        out <<= 2
        out += 0b01 if self.turns % 2 == 0 else 0b10
        
        return out << 8
    
    def expect_next(self):
        # commenting this since it doesn't change output
        # but the following is the parallel to the 'win' and 'tie' methods:
        # out = 0b0000
        # out <<= 18
        out = self.board_to_bits() << 2
        out += 0b01 if self.turns % 2 == 0 else 0b10
        out <<= 3
        out += self.turns - 1

        return out << 5

    def err(self, e):
        # self.board.print()
        out = 0
        if e == OccupiedSpaceException:
            out = 0b0100 << 20
        else:
            out = 0b0101 << 20

        out += 0b01 if self.turns % 2 == 0 else 0b10
        out <<= 3
        out += self.turns - 1
        
        return out << 5

    def board_to_bits(self):
        return self.board.to_bits()

#########################################################################################
# Neural network output style, consists of a single integer (32 bits). 
# Specification is as follows:
# ------------------------------------------------------------------
# |  CODE  |   Board State  | Player | Turn Count |    Reserved    | 
# |  4-bit |      18-bit    | 2-bit  |    3-bit   |      5-bit     |
# |        | (2 per square) |        |            | (not used yet) |
# ------------------------------------------------------------------
# Possible CODEs:
#   - 0000:      OK , expects a move
#   - 0001:      OK , game ended, tie
#   - 0010:      OK , game ended, 'o' wins
#   - 0011:      OK , game ended, 'x' wins
#   - 0100:      ERR, game ended, attempted to place 'x' or 'o' in occupied spot
#   - 0101-1111: ERR, reserved for later
#   
# Board State:
#   - 00: square is empty
#   - 10: 'x' placed
#   - 01: 'o' placed
#
#   This is done for every square, then we concatenate the bits to arrive at the
#   18-bit representation of the board. Concatenation is done from left to right,
#   then top to bottom.
#
# Ex.:
#
#   x |   | o   -->   10 | 00 | 01  -->  100001
#  -----------      -------------- 
#   o |   |     -->   01 | 00 | 00  -->  010000  -->  100001010000011010
#  -----------      --------------   
#   o | x | x   -->   01 | 10 | 10  -->  011010                 
#
# Player: 
#   Indicates if move is to be made by 'x' or 'o'
#   - 10: 'x'
#   - 01: 'o'
#   
# Turn count:
#   Indicates the number of turns taken - 1 (not including erroneous placements). 
#   Used by the genetic algorithm to calculate punishment for erroneous placements.
#   If a full game is played, this is ignored, so we only need 3 bits for this.
#
# Reserved:
#   The last 5 bits are reserved in case we see a need for them later.
#
#
# Interpretation of the data frame:
#   The first 4 bits decide the next step in the training process.
#   - An error is indicative of the network making a mistake. In this case
#     we will punish the network to decrease it's opportunity to affect the
#     next generation's offspring.
#   - An OK code tells the game runner whether to reward the network, or ask 
#     for a next move.
#
#   The board state and the 'Player' bits completely define the network's 
#   behavior. We expect that the 20 bits will comprise the 20 input neurons
#   for the network.
#########################################################################################

# this is human readable output. used for testing
def h_out(r, game):
    if   r == "state":
        game.board.print()
    elif r == "input":
        game.board.print()
        print(f"where should {game.curr_symbol()} place?")
    elif r == "win":
        game.board.print()
        print(f"{game.curr_symbol()} won")
    elif r == "tie":
        game.board.print()
        print("tie")
