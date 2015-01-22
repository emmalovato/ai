#!/usr/bin/env python
#coding:utf-8

from random import randint
from BaseAI import BaseAI
import pdb
import math
from AlphaBeta import AlphaBeta
from Grid import Grid

class PlayerAI(BaseAI):

	def getMove(self, grid):

		a = AlphaBeta()

		moves = grid.getAvailableMoves()

		maxAlpha = float("-inf")
		maxMove = -1
		for move in moves:
			gridCopy = grid.clone()
			gridCopy.move(move)
			alpha = a.alphabeta(gridCopy, 4, float("-inf"), float("inf"), True) # Call alphabeta as Max, what does this return?
			if alpha > maxAlpha:
				maxAlpha = alpha
				maxMove = move

		# it should return the move corresponding to the node with the highest minimax value
		# move should be returned in the form of an int: 0, 1, 2, 3 --> up, down, left, right
		
		return maxMove







