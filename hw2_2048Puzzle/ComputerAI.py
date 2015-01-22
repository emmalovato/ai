#!/usr/bin/env python
#coding:utf-8

from random import randint
from BaseAI import BaseAI
import pdb
from AlphaBeta import AlphaBeta
from Grid import Grid

class ComputerAI(BaseAI):


	def getMove(self, grid):

		# place a random tile in a random cell
		# cells = grid.getAvailableCells()
		# return cells[randint(0, len(cells) - 1)] if cells else None

		a = AlphaBeta()

		moves = grid.getAvailableCells()

		minBeta = float("inf")
		bestCell = (-1,-1)
		for cell in moves:
			gridCopy = grid.clone()
			# try inserting a 2 in the cell
			gridCopy.insertTile(cell, 2)
			beta1 = a.alphabeta(gridCopy, 2, float("-inf"), float("inf"), False)
			gridCopy = grid.clone()
			# try inserting a 4 in the cell
			gridCopy.insertTile(cell, 4)
			beta2 = a.alphabeta(gridCopy, 2, float("-inf"), float("inf"), False)
			# calculate the expected value of beta for each cell 
			beta = beta1*0.9 + beta2*0.1
			if beta < minBeta:
				minBeta = beta
				bestCell = cell
		
		return bestCell





