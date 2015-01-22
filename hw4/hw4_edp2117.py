#!/usr/bin/env python
#coding:utf-8

import pdb
import sys
import time
from copy import deepcopy

ROW = "ABCDEFGHI";
COL = "123456789";

# utility function to print each sudoku
def printSudoku(sudoku):
	print "-----------------"
	for i in ROW:
		for j in COL:
			print sudoku[i + j],
		print ""	


# Reading of sudoku list from file
try:
    f = open("sudokus.txt", "r")
    sudokuList = f.read()
except:
	print "Error in reading the sudoku file."
	exit()



# formulation for a constraint satisfaction problem 
# includes a list of variables, a dictionary of those variables and their corresponding domains, and the constraint for the problem
class CSP:
	# initialize csp class
	def __init__(self, X, D):
		self.X = X # X is a set of variables
		self.D = D # D is a set of domains
		self.C = self.constraint
	# takes two values, returns true if they're consistent according to the nonequal constraint and false otherwise 
	def constraint(self, x, y): # checks if a cell's domain is different from that of its neighbor via the current arc
		if x == y: # none of the values in the cells' domains can match 
			return False # if one does match, the cells are not arc consistent and they fail the constraint 
		else:
			return True

def initQueue(csp):
	queue = []
	# initialize the queue to contain all arc constraints in the problem
	# each cell has an arc between every cell in its ROW, COLUMN, and small BOX
	for cell in csp.X: # iterate through all cells in the puzzle
		# add arcs between current cell and all cells in the same row
		# first find which row the cell is in
		pos = csp.X.index(cell)
		diff = pos%9
		start = pos - diff # starts at the first cell in that row 
		end = start + 9
		while start < end:
			neighbor = csp.X[start]
			if cell != neighbor:
				if queue.count((cell, neighbor)) == 0: # only add new arcs to the queue
					queue.append((cell, neighbor))
			start += 1
		# add arcs between current cell and all cells in the same column
		start = pos%9
		# remainder 0 means column 1, r 1 column 2, ... r 8 column 9
		while start < 81:
			neighbor = csp.X[start]
			if cell != neighbor:
				if queue.count((cell, neighbor)) == 0:
					queue.append((cell, neighbor))
			start += 9
		# add arcs between current cell and all arcs in the same box
		# break into groups of 3 rows 
		if pos < 27: # A B or C
			if pos%9 < 3: # first box
				i = 0 # i represents the first cell in each row (of the box)
				while i <= 20:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
			elif pos%9 < 6: # second box
				i = 3 # i represents the first cell in each row (of the box)
				while i <= 23:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
			elif pos%9 < 9: # third box
				i = 6 # i represents the first cell in each row (of the box)
				while i <= 26:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
		elif pos < 54: # D E or F
			if pos%9 < 3: # fourth box
				i = 27 # i represents the first cell in each row (of the box)
				while i <= 47:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
			elif pos%9 < 6: # fifth box
				i = 30 # i represents the first cell in each row (of the box)
				while i <= 50:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
			elif pos%9 < 9: # sixth box
				i = 33 # i represents the first cell in each row (of the box)
				while i <= 53:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
		elif pos >= 54: # G H or I
			if pos%9 < 3: # seventh box
				i = 54 # i represents the first cell in each row (of the box)
				while i <= 74:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
			elif pos%9 < 6: # eighth box
				i = 57 # i represents the first cell in each row (of the box)
				while i <= 77:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9
			elif pos%9 < 9: # ninth box
				i = 60 # i represents the first cell in each row (of the box)
				while i <= 80:
					for j in xrange(3):
						neighbor = csp.X[i + j]
						if cell != neighbor:
							if queue.count((cell, neighbor)) == 0:
								queue.append((cell, neighbor))
					i += 9

	return queue


def getNeighbors(cell, csp):
	queue = [] # queue of arcs
	pos = csp.X.index(cell)
	diff = pos%9
	start = pos - diff # starts at the first cell in that row 
	end = start + 9
	while start < end:
		neighbor = csp.X[start]
		if cell != neighbor:
			if queue.count((cell, neighbor)) == 0: # only add new arcs to the queue
				queue.append((cell, neighbor))
		start += 1
	# add arcs between current cell and all cells in the same column
	start = pos%9
	# remainder 0 means column 1, r 1 column 2, ... r 8 column 9
	while start < 81:
		neighbor = csp.X[start]
		if cell != neighbor:
			if queue.count((cell, neighbor)) == 0:
				queue.append((cell, neighbor))
		start += 9
	# add arcs between current cell and all arcs in the same box
	# break into groups of 3 rows 
	if pos < 27: # A B or C
		if pos%9 < 3: # first box
			i = 0 # i represents the first cell in each row (of the box)
			while i <= 20:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
		elif pos%9 < 6: # second box
			i = 3 # i represents the first cell in each row (of the box)
			while i <= 23:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
		elif pos%9 < 9: # third box
			i = 6 # i represents the first cell in each row (of the box)
			while i <= 26:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
	elif pos < 54: # D E or F
		if pos%9 < 3: # fourth box
			i = 27 # i represents the first cell in each row (of the box)
			while i <= 47:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
		elif pos%9 < 6: # fifth box
			i = 30 # i represents the first cell in each row (of the box)
			while i <= 50:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
		elif pos%9 < 9: # sixth box
			i = 33 # i represents the first cell in each row (of the box)
			while i <= 53:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
	elif pos >= 54: # G H or I
		if pos%9 < 3: # seventh box
			i = 54 # i represents the first cell in each row (of the box)
			while i <= 74:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
		elif pos%9 < 6: # eighth box
			i = 57 # i represents the first cell in each row (of the box)
			while i <= 77:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9
		elif pos%9 < 9: # ninth box
			i = 60 # i represents the first cell in each row (of the box)
			while i <= 80:
				for j in xrange(3):
					neighbor = csp.X[i + j]
					if cell != neighbor:
						if queue.count((cell, neighbor)) == 0:
							queue.append((cell, neighbor))
				i += 9

	return queue

# checks to see if every cell has only one value in its domain, which for ac3 would be a complete assignment
def checkBoard(csp):
	for cell in csp.X:
		if len(csp.D[cell]) > 1:
			return False
	return True

# AC3 algorithm	
# returns false if an inconsistency is found and true otherwise 
# it accepts a binary csp with components (X, D, C)
def ac3(csp):
	failure = False # an indicator that becomes True when ac3 has failed to find a consistent assignment

	queue = initQueue(csp) # a queue of arcs, initially all the arcs in csp


	while queue:
		# remove the first arc in the queue
		(x_i, x_j) = queue.pop(0)
		if revise(csp, x_i, x_j, failure): # only update the queue further if the domain of the variable was revised
			if failure:
				return False
			neighbors = getNeighbors(x_i, csp)
			for (_,x_k) in neighbors: # get all cells with arcs pointing to the cell with revised domain
				queue.append((x_k, x_i)) # add EVERY arc pointing to that cell back to the queue
	
	return True

# checks domains and constraints, removes an x value if there are no y values that make that x value possible
# returns true if the domain of x_i was altered, false otherwise 
def revise(csp, x_i, x_j, failure):
	revised = False
	inconsistent = True
	D_i = csp.D[x_i]
	D_j = csp.D[x_j]
	if len(D_i) == 1: # if you only have one element in the domain, you just move on (that's the 'assignment')
		return False
	for x in D_i: # for every value in the domain D_i of the cell x_i
		# if no value y in D_j allows (x,y) to satisfy the constraint between x_i and x_j, delete the value x from D_i
		# assume inconsistency until a possible pair is found 
		inconsistent = True
		for y in D_j:
			# if EVERY y value fails for one x value, delete that x value 
			if csp.constraint(x,y): # if they're consistent with the constraint
				# x does not need to be removed as long as at least one consistent pair can be found in D_j's domain
				inconsistent = False
		if inconsistent == True:
			D_i.remove(x)
			if not D_i: # if the domain of any cell is empty then ac3 has failed to find a consistent assignment
				failure = True
			revised = True
	return revised



# 1.5 count number of sudokus solved by AC-3
num_ac3_solved = 0
count = 0
print "Trying to solve sudoku puzzles with AC-3..."
startTime = time.time()
for line in sudokuList.split("\n"): # each puzzle is its own csp
	# Parse sudokuList to individual sudoku in dict, e.g. sudoku["A2"] = 1
	sudoku = {ROW[i] + COL[j]: int(line[9*i+j]) for i in range(9) for j in range(9)}

	if count == 250:
		print "Halfway there!"

	if count == 400:
		print "Almost finished."

	# initialize csp
	# generate list of cell variables 
	X = [] # X is a list of all cells
	for i in xrange(len(ROW)):
		for j in xrange(len(COL)):
			var = ROW[i] + COL[j] 
			X.append(var)
	D = {} # D is a dictionary of all cells and their initial domains 
	# initialize each domain
	for i in xrange(len(ROW)*len(COL)):
		if sudoku[X[i]] is 0:
			D[X[i]] = [1,2,3,4,5,6,7,8,9]
		else: 
			D[X[i]] = [sudoku[X[i]]] # a pre-assigned variable has only its assigned value in its domain
	csp = CSP(X, D)

	if ac3(csp) and checkBoard(csp): # call ac3 function for each puzzle
		num_ac3_solved += 1

	# if the puzzle was solved, num_ac3_solved will have been incremented by one
	# if it was not solved, ac3 will return false and try the next puzzle 

	count += 1

endTime = time.time()
runtime = endTime - startTime
print "Number of puzzles solved: " + str(num_ac3_solved)
print "Runtime: " + str(runtime)

print ''
print "Now solving puzzles by backtracking..."




# backtracking search, updates the actual sudoku board when a variable is assigned a value 
def backtracking_search(csp):
	b = backtrack({}, csp)
	return b


def backtrack(assignment, csp):
	# the assignment is complete if every variable has one value in its domain 
	if isComplete(assignment, csp):
		return assignment
	# select unassigned variable by means of minimum remaining values heuristic 
	var = select(csp, assignment)
	# domain values ordered from smallest to greatest
	values = csp.D[var]
	# create deep copy of csp
	rootcsp = deepcopy(csp)
	for val in values:
		# reassign csp
		csp = deepcopy(rootcsp)
		if checkConstraints(var, val, csp):
			assignment[var] = val # add the variable to the assignment 
			# when a variable is assigned, apply forward checking to reduce the variables' domains
			forwardcheck(var, val, csp)
			result = backtrack(assignment, csp)
			if not isFailure(result, csp):
				return result
			# pruning the tree, backtracking to a different subtree 
			del assignment[var]
	# an empty dict represents a failure
	return {}

def isFailure(result, csp):
	# checks whether a resulting complete assignment of backtrack is a failure
	for cell in csp.X:
		if result == None: # if the result is null
			return True
		if cell not in result: # if it missed any cells
			return True
		if result[cell] == 0: # if any cells' values are still 0
			return True
	return False # not a failure


# checks to see if an assignment could satisfy the constraints
def checkConstraints(var, val, csp):
	x = val
	n = getNeighbors(var, csp)
	for (_,neighbor) in n:
		for y in csp.D[neighbor]:
			if csp.constraint(x,y): # if any consistent pair can be found between x and the values of y
				return True
	return False

# for each unassigned variable Y that is connected to X by a constraint, delete from Y's domain any value...
# ... that is inconsistent with the value chosen for X (this is the description of the algorithm from the book)
def forwardcheck(X, x, csp):
	n = getNeighbors(X, csp)
	for (_,neighbor) in n:
		if neighbor not in assignment: # only modify domains of unassigned variables
			d = csp.D[neighbor]
			for y in d:
				if not csp.constraint(x,y): # remove inconsistent values
					csp.D[neighbor].remove(y)



# select the variable with the smallest domain 
# in the case of ties, the last tied value will be chosen
def select(csp, assignment):
	minsize = 9
	mincell = None
	for cell in csp.X:
		size = len(csp.D[cell])
		if size < minsize and cell not in assignment: # it should only select unassigned cells
			minsize = size
			mincell = cell
	return mincell

# complete if every variable is assigned only one value
# and if it's a consistent assignment (or should this be in isFailure?)
def isComplete(assignment, csp):
	for cell in csp.X:
		if cell not in assignment:
			return False
		if not check(cell, assignment, csp):
			return False
	return True


def check(X, assignment, csp):
	n = getNeighbors(X, csp)
	x = assignment[X]
	for (_,neighbor) in n:
		if neighbor in assignment:
			y = assignment[neighbor]
			if not csp.constraint(x, y): # only a consistent assignment if every arc is consistent
				return False
	return True


# 1.6 solve all sudokus by backtracking
startTime = time.time()
for line in sudokuList.split("\n"):
	# Parse sudokuList to individual sudoku in dict, e.g. sudoku["A2"] = 1
	sudoku = {ROW[i] + COL[j]: int(line[9*i+j]) for i in range(9) for j in range(9)}
	# will contain cell:value pairs
	assignment = {} 

	# initialize csp
	# generate list of cell variables 
	X = [] # X is a list of all cells
	for i in xrange(len(ROW)):
		for j in xrange(len(COL)):
			var = ROW[i] + COL[j] 
			X.append(var)
	D = {} # D is a dictionary of all cells and their initial domains 
	# initialize each domain
	for i in xrange(len(ROW)*len(COL)):
		if sudoku[X[i]] is 0:
			D[X[i]] = [1,2,3,4,5,6,7,8,9]
		else: 
			D[X[i]] = [sudoku[X[i]]] # a pre-assigned variable has only its assigned value in its domain
	
	csp1 = CSP(X, D)

	# use ac3 to reduce domains 
	ac3(csp1)

	assignment = backtracking_search(csp1)
	for cell in assignment.keys():
		sudoku[cell] = assignment[cell]
	# print solution to each sudoku after solving it 
	printSudoku(sudoku)


endTime = time.time()
runtime = endTime - startTime
print "Runtime for backtracking: " + str(runtime)







