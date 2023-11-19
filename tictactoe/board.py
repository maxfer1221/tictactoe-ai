from common import * # common utils among the files
from collections import defaultdict


class Board:
    # initialize the board
    def __init__(self):
        # integers must differ to allow for simple win-checking function:
        #   win-check just checks if row, column, or diagonal 
        #   contains equal entries
        self.board = [
            [-1,-2,-3],
            [-4,-5,-6],
            [-7,-8,-9],
        ]

        # used for printing the board
        self.charmap = defaultdict(lambda: " ")
        self.charmap["0"] = "o"
        self.charmap["1"] = "x"
    
    # place an 'x' (1) or 'o' (-1)
    def __put(self, r, c, p):
        self.board[r][c] = p

    def put_checked(self, r, c, p):
        # position already used up
        if not self.board[r][c] < 0:
            raise OccupiedSpaceException
        
        else:
            self.__put(r, c, p)

    # check if there is a win
    def win_on_board(self):
        board = self.board
        for r in range(3):
            if board[r][0] == board[r][1] == board[r][2]:
                return True
        
        for c in range(3):
            if board[0][c] == board[1][c] == board[2][c]:
                return True

        if board[0][0] == board[1][1] == board[2][2]:
            return True

        if board[0][2] == board[1][1] == board[2][0]:
            return True
            
        return False

    def print(self):
        cm = self.charmap
        b  = self.board
        c  = lambda r,c: self.cast(f"{b[r][c]}")
        print(f' {c(0,0)} | {c(0,1)} | {c(0,2)}')
        print("-----------")
        print(f' {c(1,0)} | {c(1,1)} | {c(1,2)}')
        print("-----------")
        print(f' {c(2,0)} | {c(2,1)} | {c(2,2)}')
        print("===========")
        
    def cast(self, p):
        return self.charmap[p]

    def to_bits(self):
        bits = 0
        for r in range(3):
            for c in range(3):
                bits <<= 2
                if self.board[r][c] == 0:
                    bits += 0b01
                elif self.board[r][c] == 1:
                    bits += 0b10
        return bits
