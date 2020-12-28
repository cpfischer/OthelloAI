## OG Author: Kevin Weston
## Edit for use: Braydon Hampton & Caleb Fischer

## This program will simulate a game of othello in the console
## In order to use it, you need to:
## 1. Import code for your AIs.
## 2. You also need to import a function which applies an action to the board state,
##	and replace my function call (main.get_action(...)) within the while loop.
## 3. Your AIs must have a get_move() function. Assign these functions to
##	white_get_move and black_get_move.
## 4. Create an initial board state to be played on

## Replace these imports with the file(s) that contain your ai(s)
#import main
#import Rando
#import MiniMaximus
import random
import copy
import time
#############################################
# heatmapper.py
#############################################
import sys


## borrowed from game.py for testing
def init_board(board_size):
    board_state = [[' '] * board_size for i in range(board_size)] 

    board_state[board_size / 2][board_size / 2] = 'W'
    board_state[board_size / 2 - 1][board_size / 2 - 1] = 'W'
    board_state[board_size / 2 - 1][board_size / 2] = 'B'
    board_state[board_size / 2][board_size / 2 - 1] = 'B'

    #print_board(board_state, board_size)

    return board_state

def matrix(n, s):
    chunkContainer = []
    for i in range(0, (n * n), n):
        chunkContainer.append(s[i:i + n])
    return chunkContainer

def isCorner(n, i):
    return (i == n - 1 or i == n * n - 1 or i == (n * n) - n or i == 0)

# def print_board(board_state, board_size):
#     s = '\t '.join([str(x) for x in range(board_size)])
#     print "   ", s
#     print "--------" * board_size
#     row_num = 0
#     for row in board_state:
#         q = '\t|'.join([str(x) for x in row])
#         print row_num, '|', q
#         row_num += 1

def generate_heatmap(n):
    DEFAULT = 1
    heuristic = [DEFAULT] * (n * n)
    seperator = ','
    MAX = 1000
    
    #CORNERS========================================
    heuristic[0] = MAX
    for i in range(1, (n * n), 1):
        if (isCorner(n, i)):
            heuristic[i] = MAX

    matrixBoard = matrix(n, heuristic)

    #SECONDTOLAST=======================================================================
    SECOND_ROW = 1
    for i in range(n):
        if (matrixBoard[1][i] != MAX):
            matrixBoard[1][i] = SECOND_ROW

    # last row
    for i in range(n):
        if (matrixBoard[n - 2][i] != MAX):
            matrixBoard[n - 2][i] = SECOND_ROW

    # col 0
    for i in range(n):
        if (matrixBoard[i][1] != MAX):
            matrixBoard[i][1] = SECOND_ROW

    # last col
    for i in range(n):
        if (matrixBoard[i][n - 2] != MAX):
            matrixBoard[i][n - 2] = SECOND_ROW

    #EDGES==============================================================================
    # row 0
    EDGE_VAL = 10
    for i in range(n):
        if (matrixBoard[0][i] != MAX):
            matrixBoard[0][i] = EDGE_VAL

    # last row
    for i in range(n):
        if (matrixBoard[n - 1][i] != MAX):
            matrixBoard[n - 1][i] = EDGE_VAL

    # col 0
    for i in range(n):
        if (matrixBoard[i][0] != MAX):
            matrixBoard[i][0] = EDGE_VAL

    # last col
    for i in range(n):
        if (matrixBoard[i][n - 1] != MAX):
            matrixBoard[i][n - 1] = EDGE_VAL
    #===============================================================================
    
    #BY_CORNERS========================================================================
    #top left
    DIAG_CORNER = -10
    EDGE_CORNER = -10
    matrixBoard[0][1] = EDGE_CORNER
    matrixBoard[1][1] = DIAG_CORNER
    matrixBoard[1][0] = EDGE_CORNER

    #top right
    matrixBoard[0][n - 2] = EDGE_CORNER
    matrixBoard[1][n - 2] = DIAG_CORNER
    matrixBoard[1][n - 1] = EDGE_CORNER

    #bottom left
    matrixBoard[n - 2][0] = EDGE_CORNER
    matrixBoard[n - 2][1] = DIAG_CORNER
    matrixBoard[n - 1][1] = EDGE_CORNER
    
    #bottom right
    matrixBoard[n - 2][n - 1] = EDGE_CORNER
    matrixBoard[n - 2][n - 2] = DIAG_CORNER
    matrixBoard[n - 1][n - 2] = EDGE_CORNER
    #==========================================================================

    #print_board(matrixBoard, n)
    return matrixBoard
        
        
def eval_heur(board_state, heuristic, parent_turn):
    w_count = 0
    b_count = 0
    empty_count = 0
    w_weight = 0
    b_weight = 0
    
    for i in range(len(board_state)):
        for j in range(len(board_state)):
            if board_state[i][j] == ' ':
                empty_count += 1
            elif board_state[i][j] == 'B':
                b_count += 1
                b_weight += heuristic[i][j]
            elif board_state[i][j] == 'W':
                w_count += 1
                w_weight += heuristic[i][j]
    
    if parent_turn == 'B':
        return b_weight - w_weight
    else:
        return w_weight - b_weight

#############################################
# End of heatmapper.py
#############################################

#############################################
# ClosedList.py
#############################################
class ClosedList(object):
    def __init__(self):
        object.__init__(self)
    def put(self, x):
        """ Put x into closed list """
        raise NotImplementedError
    def __contains__(self, x):
        """ Implements "in" operator to check is x is in the closed list """
        raise NotImplementedError
    def size(self):
        """ Returns the number of values in the closed list """
        raise NotImplementedError
    def __len__(self):
        """ Returns the number of values in the closed list """
        return 0 # Must be overwritten
    def values(self):
        raise NotImplementedError
    def __iter__(self):
        raise NotImplementedError
    def clear(self):
        raise NotImplementedError

class TranspositionTable(ClosedList):
    """ Implementation of ClosedList using python sets and with state compression"""
    def __init__(self, compress=lambda x:x, decompress=lambda x:x):
        """
        compress -- a function to compress state
        decompress -- inverse function of compress
        """
        ClosedList.__init__(self)
        self.table = dict()
        self.compress = compress
        self.decompress = decompress

    def to_tuple(self, b):
        return tuple(tuple(r) for r in b)

    def add(self, state, score):
        stateTuple = self.compress(self.to_tuple(state))
        self.table[stateTuple] = score

    def __contains__(self, state):
        stateTuple = self.compress(self.to_tuple(state))
        return stateTuple in self.table
    def get_score(self, state):
        stateTuple = self.compress(self.to_tuple(state))
        return self.table[stateTuple]
    def size(self):
        return len(self.table)
    def __len__(self):
        return len(self.table)
    def __str__(self):
        s = str(self.table)[5:-2]
        return "<TranspositionTable {%s}>" % s
    def values(self):
        f = self.decompress
        return [f(x) for x in self.table]    
    def __iter__(self):
        for x in self.table:
            yield f(x)
        raise StopIteration
    def clear(self):
        self.table.clear()

        #test

#############################################
# end of ClosedList.py
#############################################

#############################################
# minimax.py
#############################################
class node:
    def __init__(self, value=None, parent=None):
        self.value = value
        self.parent = parent
        self.children = []

    def __str__(self):
        if self.parent == None:
            return "Root Node: %s" % str(id(self))
        elif self.value is not None:
            return "\tLeaf Node: %s from parent %s" % (str(self.value), str(id(self.parent)))
        else:
            return "Node: %s from parent %s" % (str(id(self)), str(id(self.parent)))

    def prints(self, level=0):
        print '\t' * level + str(self) + "  Term: %s" % self.isTerm()
        for child in self.children:
            child.prints(level+1)

    def isTerm(self):
        return len(self.children) is 0

    def populate(self, width, depth):
        if depth is not 1:
            for val in range(width):
                self.children.append(node(None, self))
                self.children[val].populate(width, depth-1)
        else:
            for val in range(width):
                x = input("Leaf node: ")
                self.children.append(node(x, self))

    def RandomPopulate(self, width, depth):
        if depth is not 1:
            for val in range(width):
                self.children.append(node(None, self))
                self.children[val].RandomPopulate(width, depth-1)
        else:
            for val in range(width):
                x = random.randint(1, 20) 
                self.children.append(node(x, self))
                
class SearchNode():
    def __init__(self, board_state, board_size, turn, move = None, depth = 0, score = None):
        self.board_state = board_state
        self.board_size = board_size
        self.turn = turn
        self.move = move
        self.score = score
        self.depth = depth

    def __str__(self):
         return '<SearchNode %s %s %s %s %s>' % (id(self),
                                                self.turn,
                                                self.move,
                                                self.score,
                                                self.depth)



def minimax_min(current_node, alpha, beta, max_depth, heuristic, transposition, corners, parent_turn):
    valid_moves = get_possible_moves(current_node.board_size, current_node.board_state, current_node.turn)
    next_turn = 'W' if current_node.turn == 'B' else 'B'
    
    if max_depth == current_node.depth or len(valid_moves) <= 0:
        current_node.score = eval_heur(current_node.board_state, heuristic, parent_turn)
        return current_node
    else:
        
        next_turn = 'W' if current_node.turn == 'B' else 'B'
        for move_flip in valid_moves:
            new_state = do_move(current_node.board_state, move_flip[1], current_node.turn)
            
            node_after_move = SearchNode(new_state, current_node.board_size, next_turn, move_flip[0], (current_node.depth + 1))
            if node_after_move.move in corners: 
                node_after_move.score = eval_heur(current_node.board_state, heuristic, parent_turn)
                return node_after_move


            if new_state in transposition:
                break
            transposition.add(new_state, None)

            leaf_node = minimax_max(node_after_move, alpha, beta, max_depth, heuristic, transposition, corners, parent_turn)
            node_after_move.score = leaf_node.score
           
            if leaf_node.score < beta.score:
                beta = node_after_move
            if beta.score <= alpha.score:
                break
        return beta

def minimax_max(current_node, alpha, beta, max_depth, heuristic, transposition, corners, parent_turn):
    valid_moves = get_possible_moves(current_node.board_size, current_node.board_state, current_node.turn)
    next_turn = 'W' if current_node.turn == 'B' else 'B'

    if max_depth == current_node.depth or len(valid_moves) <= 0:
        current_node.score = eval_heur(current_node.board_state, heuristic, parent_turn)
        return current_node
    else:
        next_turn = 'W' if current_node.turn == 'B' else 'B'
        for move_flip in valid_moves:
            new_state = do_move(current_node.board_state, move_flip[1], current_node.turn)
            
            node_after_move = SearchNode(new_state, current_node.board_size, next_turn, move_flip[0], (current_node.depth + 1))
            if node_after_move.move in corners: 
                node_after_move.score = eval_heur(current_node.board_state, heuristic, parent_turn)
                return node_after_move
                        
            if new_state in transposition:
                break
            transposition.add(new_state, None)

            leaf_node = minimax_min(node_after_move, alpha, beta, max_depth, heuristic, transposition, corners, parent_turn)
            node_after_move.score = leaf_node.score
            

            if leaf_node.score > alpha.score:
                alpha = node_after_move
            if beta.score <= alpha.score:
                break
        return alpha

#############################################
# End of minimax.py
#############################################

#############################################
# game.py
#############################################
## randyMove, userMove, init_board added by BH

def is_in_bounds(board_size, (r, c)):
    return r < (board_size) and c < (board_size) and r >= 0 and c >= 0

def do_move2(board_state, flip_list, turn):
    #new_state = copy.deepcopy(board_state)
    for space in flip_list:
        board_state[space[0]][space[1]] = turn
    #return new_state

def do_move(board_state, flip_list, turn):
    new_state = copy.deepcopy(board_state)
    for space in flip_list:
        new_state[space[0]][space[1]] = turn
    return new_state

def get_possible_moves(board_size, board_state, turn):
    valid_moves = []
    for r in range(0, board_size, 1):
        for c in range(0, board_size, 1):
            moves = get_valid_move(board_size, board_state, turn, (r,c))
            if len(moves) > 0:
                for sublist in moves:
                    valid_moves.append(sublist) #flatten list
    
    
    valid_moves_remove_dupes = {}
    returned_moves = []
    if len(valid_moves) > 0:
        for move in valid_moves:
            if move[0] in valid_moves_remove_dupes:
                index = valid_moves_remove_dupes[move[0]]
                returned_moves[index] = (returned_moves[index][0], returned_moves[index][1] + move[1]) #concat to combine two sets of flipping
            else:
                returned_moves.append(move)
                valid_moves_remove_dupes[move[0]] =  (len(returned_moves) - 1)
    #print returned_moves
    return returned_moves

def get_valid_move(board_size, board_state, turn, (r,c)): #turn = 'B' or 'W'
    opponent_tile = 'W' if turn == 'B' else 'B'
    
    valid_moves = []
    if board_state[r][c] == turn: #only search around turn tiles
        for (row_direction, col_direction) in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            (row_search,col_search) = (r,c)
            flip_list = []
            row_search = row_search + row_direction
            col_search = col_search + col_direction

            while is_in_bounds(board_size, (row_search,col_search)) and board_state[row_search][col_search] == opponent_tile:
                flip_list.append((row_search, col_search))
                row_search += row_direction
                col_search += col_direction
            
            if (
                is_in_bounds(board_size, (row_search,col_search)) 
                and len(flip_list) > 0 
                and board_state[row_search][col_search] == ' '
            ):        
                flip_list.append((row_search,col_search))                                                    #some tiles can have multiple moves
                valid_moves.append(((row_search, col_search), flip_list)) #only append if end is a space

    return valid_moves

def get_move(board_size, board_state, turn, time_left, opponent_time_left):
    time_in_seconds = time_left / 1000
    #for best move as black, maxDepth should be odd, white -> even
    timeForMove = time_in_seconds / (board_size * board_size - 4)
    #print "Time for move: %s" % timeForMove
    startTime = time.clock()
    
    corners = [(0,0), (0, board_size-1), (board_size-1, 0), (board_size, board_size)]
    # bad_spaces = [(1,0), (0,1), (1,1), 
    #              (0, board_size - 2), (1, board_size - 2), (1, board_size - 1),
    #              (board_size - 2, 0), (board_size - 2, 1), (board_size - 1, 1),
    #              (board_size - 2, board_size - 1), (board_size - 2, board_size - 2), (board_size - 1, board_size - 2)]
    
    maxDepth = 5
    possible_moves = get_possible_moves(board_size, board_state, turn)
    if len(possible_moves) is 0: 
        return None

#############################################################
        # TAKE THIS OUT BEFORE UPLOAD
        #do_move2(board_state, possible_moves[0][1], turn) 
##############################################################        
        return possible_moves[0][0]
    
    heur = generate_heatmap(board_size)
    INF = 100000000
    initialNode = SearchNode(board_state, board_size, turn, None, 0)
    
    alphaNode = SearchNode(None, None, None, None, None, -INF)
    betaNode = SearchNode(None, None, None, None, None, INF)
    transposition = TranspositionTable()

    parent_turn = turn
    bestMove = minimax_max(initialNode, alphaNode, betaNode, maxDepth, heur, transposition, corners, parent_turn)
    while (timeForMove - 1 > (time.clock() - startTime)):
        transposition.table.clear()
        maxDepth += 1
        bestMove = minimax_max(initialNode, alphaNode, betaNode, maxDepth, heur, transposition, corners, parent_turn)
        #print "%s leads to this: %s" % (bestMove.move, bestMove.score)
        if bestMove.move is None or maxDepth > 15: break
        
    ###########################################################
    # TAKE THIS BLOCK OUT BEFORE UPLOAD
    # for child in possible_moves:
    #     if child[0] == bestMove.move:
    #         do_move2(board_state, child[1], turn)
    #         break   
    ############################################################
    
    endTime = time.clock()
    #print "Time to get move: %.2f" % (endTime - startTime)
    #print "Depth reached: %s" % maxDepth

    #print "Returning from mm: %s" % str(bestMove.move)
    return bestMove.move


def init_board(board_size):
    board_state = [[' '] * board_size for i in range(board_size)] 

    board_state[board_size / 2][board_size / 2] = 'W'
    board_state[board_size / 2 - 1][board_size / 2 - 1] = 'W'
    board_state[board_size / 2 - 1][board_size / 2] = 'B'
    board_state[board_size / 2][board_size / 2 - 1] = 'B'

    #print_board(board_state, board_size)

    return board_state


def get_winner(board_state):
    black_score = 0
    white_score = 0
    for row in board_state: 
        for col in row: 
            if col == 'W':
                white_score += 1
            elif col == 'B':
                black_score += 1
    
    if black_score > white_score:
        winner = 'B'
    elif white_score > black_score:
        winner = 'W'
    else:
        winner = None
    return (winner, white_score, black_score)


def prepare_next_turn(turn, white_get_move, black_get_move, pTime, opTime):
	next_turn = 'W' if turn == 'B' else 'B'
	next_move_function = white_get_move if next_turn == 'W' else black_get_move

	return next_turn, next_move_function, opTime, pTime

# needs state and size atm - BH
# def print_board(board_state, board_size, tabover = 0):
#     tabover = '\t' * tabover
#     s = '    '.join([str(x) for x in range(board_size)])
#     print tabover, "   ", s
#     row_num = 0
#     for row in board_state:
#         print tabover, row_num, row
#         row_num += 1
   	 

def apply_action(board_state, action, turn):
    board_state[action[0]][action[1]] = turn
    return board_state

# def simulate_game(board_state,
#               	board_size,
#               	white_get_move,
#               	black_get_move):
#     player_blocked = False
#     turn = 'B'
#     get_move = black_get_move
#     white_total_time = 0.0
#     black_total_time = 0.0
#     blackTurnCount = 0
#     whiteTurnCount = 0
#     timer1, timer2 = 180.0, 180.0
#     #print_board(board_state, board_size)
#     see_moves = input("See moves? Yes = 0  No = 1")
#     while True:
#         ## GET ACTION ##
#         if timer1 < 0 or timer2 < 0: break
#         startTime = time.clock()
#         next_action = get_move(board_size=board_size,
#                                 board_state=board_state,
#                                 turn=turn, 
#                                 time_left=timer1, 
#                                 opponent_time_left=timer2)
#         endTime = time.clock()
#         timer1 -= (endTime - startTime)

#         if (next_action is not None):
#             if (turn == 'B'):
#                 black_total_time += endTime - startTime
#                 blackTurnCount = blackTurnCount + 1
#             else:
#                 whiteTurnCount = whiteTurnCount + 1
#                 white_total_time += endTime - startTime
        
#         #print "turn: ", turn, "next action: ", next_action
#         if see_moves == 0: _ = raw_input()
#         ## CHECK FOR BLOCKED PLAYER ##
#         # The gameover check - BH
#         if next_action is None:
#             if player_blocked:
#                 #print "Both players blocked!"
#                 break
#             else:
#                 player_blocked = True
#                 #print "Player %s had to skip!" % str(turn)
#                 time_left = 0
#                 opponent_time_left = 0
#                 turn, get_move, timer1, timer2 = prepare_next_turn(turn, white_get_move, black_get_move, timer1, timer2)
#                 continue
#         else:
#             player_blocked = False

#         #print_board(board_state, board_size)
#         turn, get_move, timer1, timer2 = prepare_next_turn(turn, white_get_move, black_get_move, timer1, timer2)

# 	winner, white_score, black_score = get_winner(board_state)

#     #print "Total game time: %.2f" % (black_total_time + white_total_time)

#     #print "\nGame stats for BLACK: \n\tScore: %s\n\tTotal time: %s\n\tTime Left: %s\n\t# Moves: %s\n\tAvg time/turn: %s" % (black_score, black_total_time, timer1, blackTurnCount, (black_total_time / blackTurnCount))
#     #print "\nGame stats for WHITE: \n\tScore: %s\n\tTotal time: %s\n\tTime Left: %s\n\t# Moves: %s\n\tAvg time/turn: %s" % (white_score, white_total_time, timer2, whiteTurnCount, (white_total_time / whiteTurnCount))

# # if __name__ == "__main__":
	
# #     board_size = board_size = input("Board Size: ")
# #     while board_size % 2 is not 0:
# #         #print "Invalid board size"
# #         board_size = input("Board Size: ")

# #     board_state = init_board(board_size)

# # 	## Give these the get_move functions from whatever ais you want to test
# #     white_get_move = get_move
# #     black_get_move = get_move
# #     simulate_game(board_state, board_size, white_get_move, black_get_move)

#############################################
# end of game.py
#############################################