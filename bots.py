#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from queue import Queue

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def heuristic(self, asp, board, loc, opploc):
        rows = 17
        bfmap1=[]
        for row in range(rows): bfmap1 += [[1000]*rows]
        prevmap1 = {}
        visited1 = {}
        bfq = Queue()
        bfq.put(loc)
        prevmap1[loc] = 0
        visited1[loc] = 1
        while bfq.empty() is False:
            curr = bfq.get()
            (r, c) = curr
            bfmap1[r][c] = prevmap1[curr]
            possibilities = list(asp.get_safe_actions(board, curr))
            for action in possibilities:
                nextloc = asp.move(curr, action)
                if nextloc not in visited1:
                    prevmap1[nextloc] = prevmap1[curr] + 1
                    bfq.put(nextloc)
                    visited1[nextloc] = 1
                

        bfmap2=[]
        for row in range(rows): bfmap2 += [[1000]*rows]
        prevmap2 = {}
        visited2 = {}
        prevmap2[opploc] = 0
        visited2[opploc] = 1
        bfq.put(opploc)

        while bfq.empty() is False:
            curr = bfq.get()
            r, c = curr
            bfmap2[r][c] = prevmap2[curr]
            possibilities = list(asp.get_safe_actions(board, curr))
            if not possibilities: 
                continue
            for action in possibilities:
                nextloc = asp.move(curr, action)
                if nextloc not in visited2:
                    prevmap2[nextloc] = prevmap2[curr] + 1
                    bfq.put(nextloc)
                    visited2[nextloc] = 1

        a1 = np.array(bfmap1)
        a2 = np.array(bfmap2)

        farray = np.subtract(a1, a2)

        farray = farray.tolist()

        poscounter = 0
        negcounter = 0
        zcounter = 0


        for i in range(len(farray)):
            for j in range(len(farray[0])):
                if farray[i][j] > 0:
                    poscounter += 1
                if farray[i][j] < 0:
                    negcounter += 1
                else:
                    zcounter += 1

        return negcounter - poscounter

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        next_ptm = (ptm + 1) % 2

        loc = locs[ptm]
        opploc = locs[next_ptm]

    
        value = float("-inf")
        beta = float("inf")
        alpha = float("-inf")
        best_action = None
        counter = 0
        eval_func = self.heuristic
        cut = 4

        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[ptm]
        
        for action in list(asp.get_safe_actions(board, loc)):
            next_state = asp.transition(state, action)
            min_val = self.abcmin(asp, next_state, alpha, beta, counter, cut, eval_func, ptm)
            if min_val > value:
                value = min_val
                best_action = action
            if value >= beta:
                return best_action
            alpha = max(alpha, value)
        return best_action
            


    def abcmax(self, asp, state, alpha, beta, counter, cut, eval_func, ptm):
        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[ptm]
        counter += 1
        loc = state.player_locs[ptm]
        board = state.board
        next_ptm = (ptm + 1) % 2
        opploc = state.player_locs[next_ptm]
        if counter == cut:
            return eval_func(asp, board, loc, opploc)
        value = float('-inf')
        for action in list(asp.get_safe_actions(board, loc)):
            next_state = asp.transition(state, action)
            value = max(value, self.abcmin(asp, next_state, alpha, beta, counter, cut, eval_func, ptm))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def abcmin(self, asp, state, alpha, beta, counter, cut, eval_func, ptm):
        if asp.is_terminal_state(state):
            return asp.evaluate_state(state)[ptm]
        counter += 1
        loc = state.player_locs[ptm]
        board = state.board
        next_ptm = (ptm + 1) % 2
        opploc = state.player_locs[next_ptm]
        if counter == cut:
            return eval_func(asp, board, opploc, loc)
        value = float('inf')
        for action in list(asp.get_safe_actions(board, opploc)):
            next_state = asp.transition(state, action)
            value = min(value, self.abcmax(asp, next_state, alpha, beta, counter, cut, eval_func, ptm))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass



class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision