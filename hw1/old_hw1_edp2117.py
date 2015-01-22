
import argparse
parser = argparse.ArgumentParser(description='Robot Path Planing | HW 1 | COMS 4701')
parser.add_argument('-bfs', action="store_true", default=False, help="Run BFS on the map")
parser.add_argument('-dfs', action="store_true", default=False, help="Run DFS on the map")
parser.add_argument('-astar', action="store_true", default=False, help="Run A* on the map")
parser.add_argument('-all', action="store_true", default=False, help="Run all 3 of the algorithms on the map")
parser.add_argument('-m', action="store", help="Map filename")

results = parser.parse_args()

if results.m=="" or not(results.all or results.astar or results.bfs or results.dfs):
    print "Check the parameters : >> python hw1_UNI.py -h"
    exit()

if results.all:
    results.bfs = results.dfs = results.astar = True

# Read the map given, initialize arena
try:
    with open(results.m) as f:
        arena = f.read()
        arena = arena.split("\n")[:-1]
except:
    print "Error in reading the arena file."
    exit() 

# Internal representation of arena
print arena

print "The arena is size " + str(len(arena)) + "x" + str(len(arena[0]))
print "\n".join(arena)

# Node class
class Node:
    val = ""
    visited = false
    left = Node()
    right = Node()  
    up = Node()
    down = Node()
    def __init__(self, val):
        self.val = val


# Find the starting position and goal position 
rows = len(arena)
columns = len(arena[0])
start = []
goal = []
for i in range(rows):
    for j in range(columns):
        if arena[i][j] == 's':
            start.append(i)
            start.append(j)
            print "start: "
            print start
        if arena[i][j] == 'g':
            goal.append(i)
            goal.append(j)
            print "goal: "
            print goal

# Breadth First Search
def bfs(arena, start, goal):
    f = []
    f.append(start) 
     

if results.bfs:
    # call BFS 
    print "BFS algorithm called"
    # write results to output file

# Depth First Search
if results.dfs:
    # call DFS
    print "DFS algorithm called"    
    #write result to output file

# A* Search
if results.astar:
    # call A*
    print "A* algorithm called"
    # write result to output file


