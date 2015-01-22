__author__ = "Ilambharathi Kanniah"
__email__ = "ik2342@columbia.edu"

import argparse
parser = argparse.ArgumentParser(description='Robot Path Planning | HW 1 | COMS 4701')
parser.add_argument('-bfs', action="store_true", default=False , help="Run BFS on the map")
parser.add_argument('-dfs', action="store_true", default=False, help= "Run DFS on the map")
parser.add_argument('-astar', action="store_true", default=False, help="Run A* on the map")
parser.add_argument('-all', action="store_true", default=False, help="Run all the 3 algorithms")
parser.add_argument('-m', action="store", help="Map filename")

results = parser.parse_args()

if results.m=="" or not(results.all or results.astar or results.bfs or results.dfs):
	print "Check the parameters : >> python hw1_UNI.py -h"
	exit()

if results.all:
	results.bfs = results.dfs = results.astar = True

# Reading of map given and all other initializations
try:
    with open(results.m) as f:
        arena = f.read()
        arena = arena.split("\n")[:-1]
except:
	print "Error in reading the arena file."
	exit()
  
# Internal representation  
print arena 

print "The arena of size "+ str(len(arena)) + "x" + str(len(arena[0]))
print "\n".join(arena)

if results.bfs:
	# Call / write your BFS algorithm
	print "BFS algorithm called"  # comment out later

if results.dfs:
	# Call / write your DFS algorithm
	print "DFS algorithm called"  # comment out later

if results.astar:
	# Call / write your A* algorithm
	print "A* algorithm called"  # comment out later

