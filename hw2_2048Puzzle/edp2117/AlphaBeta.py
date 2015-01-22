from Grid import Grid
from BaseAI import BaseAI
import pdb
from random import randint
import math

class AlphaBeta():


	def evaluation(self, grid):
		neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
		totalScore = 0
		# iterate through grid only once, otherwise it's too slow!
		for i in xrange(4):
			for j in xrange(4):
				cellVal = grid.getCellValue((i,j))
				# sum board tiles, add to score
				totalScore += cellVal
				pos = (i,j)
				maxTile = grid.getMaxTile()
				# bonus if maxtile is on an edge
				if cellVal == maxTile or cellVal >= maxTile/2:
					if pos[0] == 0 or pos[0] == 4 or pos[1] == 0 or pos[1] == 4:
						totalScore += 8*maxTile
					else:
						totalScore -= 6*maxTile
				# subract points for the difference between each tile and its adjacent tile 
				for neighbor in neighbors:
					newPos = (pos + neighbor)
					if newPos[0] <= 4 and newPos[0] >= 0 and newPos[1] <= 4 and newPos[1] >= 0:
						nVal = grid.getCellValue((newPos[0],newPos[1]))
						diff = math.fabs(cellVal - nVal)
						totalScore -= 2*diff*diff
					else:
						continue

		# add points for large scoring tiles in the corners
		corners = [(0,0), (0,4), (4,0), (4,4)]
		for corner in corners:
			val = grid.getCellValue(corner)
			if val == maxTile:
				totalScore += 10*maxTile
			elif val >= maxTile/2:
				totalScore += 5*maxTile
			else:
				totalScore -= 4*maxTile

		# take away points for too few free tiles 
		freeTiles = grid.getAvailableCells()
		amt = len(freeTiles)
		if amt <= 3:
			totalScore -= 18*(4-amt)
		if amt > 3 and amt < 7:
			totalScore -= 6*(7-amt)
		# add points for a lot of free tiles 
		if amt >= 7:
			totalScore += 8*amt

		return totalScore

	def hasChildren(self, grid, maxPlayer): 
		playerMoves = grid.getAvailableMoves()
		compMoves = grid.getAvailableCells()
		if maxPlayer and len(playerMoves) == 0: # if player has no available moves
			return False
		elif maxPlayer == False and len(compMoves) == 0: # if computer has no available moves (will this ever happen?)
			return False
		else:
			return True


	def alphabeta(self, grid, depth, alpha, beta, maxPlayer):
		self.grid = grid
		leaf = not self.hasChildren(grid, maxPlayer) # the node is a leaf if it doesn't have children
		if depth == 0 or leaf == True: 
			hVal = self.evaluation(grid)
			return hVal
		if maxPlayer:
			# generate possible moves from that node 
			moves = grid.getAvailableMoves()
			for move in moves: # loop on all possible moves instead
				gridCopy = grid.clone()
				# each move gets a new grid state
				gridCopy.move(move)
				value = self.alphabeta(gridCopy, depth - 1, alpha, beta, False)
				if alpha < value:
					alpha = value
				if beta <= alpha:
					break
			return alpha
		else:
			moves = grid.getAvailableCells()
			for cell in moves: # iterate on all possible moves  
				gridCopy = grid.clone()
				# generate the grid resulting from the current move
				#gridCopy.insertTile(cell, 2)
				number = randint(0,9)
				if number < 9:
					gridCopy.insertTile(cell, 2) #add 4
				if number == 9:
					gridCopy.insertTile(cell, 4)
				value = self.alphabeta(gridCopy, depth - 1, alpha, beta, True)
				if beta > value:
					beta = value
				if beta <= alpha:
					break
			return beta
		# make sure that the grid in the right state when this function returns! 


