import sys
import argparse
import Queue
import time
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


# Create output file
try:
    num = results.m[5]
    filename = "hw1_output" + num + ".txt"
    f = open(filename, 'w')
except:
    print "Error in creating output file"
    exit()

# Internal representation of arena
print arena

print "The arena is size " + str(len(arena)) + "x" + str(len(arena[0]))
print "\n".join(arena)

# Node class
class Node:
    def __init__(self, val, pos):
        self.val = val
        self.pos = pos
        self.visited = False
    def __str__(self):
        return "%s" % list(self.pos)

# Build graph as a list of lists of nodes
# set value and position of each node according to the arena
rows = len(arena)
columns = len(arena[0])
graph = []
pos = []
for i in xrange(rows):
    graph.append([])
    for j in xrange(columns):
        if type(arena[i][j]) is str:
            pos.append(i)
            pos.append(j)
            graph[i].append(Node(arena[i][j], pos))
            pos = []

# Print graph to console
#for i in xrange(rows):
#    sys.stdout.write("\n")
#    for j in xrange(columns):
#        sys.stdout.write(str(graph[i][j]))
#print ""

# Determine the starting state
pos = []
start = Node('s', pos)
for i in xrange(rows):
    for j in xrange(columns):
        if graph[i][j].val == 's':
            start.pos.append(i)
            start.pos.append(j)
            print "start:"
            print start.pos

# Determine the goal state
pos = []
goal = Node('g', pos)
for i in xrange(rows):
    for j in xrange(columns):
        if graph[i][j].val == 'g':
            goal.pos.append(i)
            goal.pos.append(j)
            print "goal:"
            print goal.pos

# Heuristic for A* algorithm
# Estimates the "manhattan distance" between two points on the grid
def dist(node, goal):
    rowDiff = abs(node.pos[0] - goal.pos[0])
    columnDiff = abs(node.pos[1] - goal.pos[1])
    #print "dist: " + str(rowDiff + columnDiff)
    return rowDiff + columnDiff

# Priority Queue for A* algorithm
class PriorityQueue:
    def __init__(self):
        self.q = list()

    def put(self, item):
        data, priority = item
        #print "Priority: " + str(priority)
        #print len(data)
        self._insort_right((priority, data))

    def get(self):
        # print type(self.q[0][1])
        return self.q.pop(0)[1] # returns the first path in the queue

    # sort according to f cost such that the first path has the lowest f cost 
    def _insort_right(self, x):
        lo = 0
        hi = len(self.q)
        while lo < hi:
            mid = (lo+hi)/2
            if x[0] < self.q[mid][0]:
                hi = mid
            else:
                lo = mid+1
        self.q.insert(lo, x)

# set all nodes in the graph to 'unvisited'
def resetGraph(graph):
    for i in xrange(rows):
        for j in xrange(columns):
            graph[i][j].visited = False
            graph[i][j].val = arena[i][j]

# returns a string containing graph with final path traced by * characters
def writeMap(path):
    finalMap = ''
    # trace path
    for i in xrange(1, len(path)-1):
        pos = path[i].pos 
        graph[pos[0]][pos[1]] = Node('*', pos)
    for i in xrange(rows):
        for j in xrange(columns):
            finalMap += str(graph[i][j].val)
        finalMap += "\n"
    return finalMap

# Breadth First Search
def bfs(graph, start):
    # fringe implemented as a FIFO list of paths (behaves like a queue)
    fringe = []
    fringe.append([start])
    #start.visited = True
    #graph[start.pos[0]][start.pos[1]].visited = True
    explored = [start]
    moves = [(0,-1),(-1,0),(0,1),(1,0)] # left, up, right, down
    #print "fringe length" + str(len(fringe))
    while fringe:
        #for f in fringe:
        #    print len(f)

        # get the first path from the fringe queue
        path = fringe.pop(0)
        # get the last node from the path
        node = path[-1]
        # get all adjacent nodes, construct a new path and push it to the fringe queue
        pos = node.pos
        #print "position: " + str(pos)
        for i in xrange(len(moves)):
            if not (0 <= pos[0] + moves[i][0] < rows and 0 <= pos[1] + moves[i][1] < columns):
                continue
            neighbor = graph[pos[0] + moves[i][0]][pos[1] + moves[i][1]]
            # avoid obstacles
            if neighbor.val == 'o':
                continue
            #if not neighbor.visited:
            if explored.count(neighbor) == 0:
                explored.append(neighbor)
                # copy path by value
                newPath = path[:]
                newPath.append(neighbor)
                # goal check
                if neighbor.val == 'g':
                    print "PATH FOUND!!"
                    #for n in newPath:
                    #    print str(n)
                    print "Nodes explored: " + str(len(explored))
                    return newPath
                fringe.append(newPath)    

def dfs(graph, start):
     # fringe implemented as a LIFO list (behaves like a stack)
    fringe = []
    fringe.append([start])
    explored = [start]
    #start.visited = True
    #graph[start.pos[0]][start.pos[1]].visited = True
    moves = [(0,-1),(-1,0),(0,1),(1,0)] # left, up, right, down
    while fringe:
        # get the last path on the fringe stack
        path = fringe.pop()
        # get the last node from the path
        node = path[-1]
        # get all adjacent nodes, construct a new path and push it to the fringe queue
        pos = node.pos
        for i in xrange(len(moves)):
            if not (0 <= pos[0] + moves[i][0] < rows and 0 <= pos[1] + moves[i][1] < columns):
                continue
            neighbor = graph[pos[0] + moves[i][0]][pos[1] + moves[i][1]]
            if neighbor.val == 'o':
                explored.append(neighbor)
            # only expand nodes that haven't already been explored
            if explored.count(neighbor) == 0:
                explored.append(neighbor)
                newPath = path[:]
                newPath.append(neighbor)
                if neighbor.val == 'g':
                    print "PATH FOUND!!"
                    print "Nodes explored: " + str(len(explored))
                    return newPath
                fringe.append(newPath) 


def astar(graph, start, goal):
    # fringe a priority queue sorted by distance from goal
    fringe = PriorityQueue()
    fcost = 0
    gcost = 0
    hcost = 0
    explored = {start:gcost}
    fringe.put(([start], fcost))
    start.visited = True
    graph[start.pos[0]][start.pos[1]].visited = True
    moves = [(0,-1),(-1,0),(0,1),(1,0)] # left, up, right, down
    while len(fringe.q) > 0:
        # get the first path on the fringe 
        # this path will be the one on the fringe with the lowest f cost 
        path = fringe.get()
        # get the last node from the path
        node = path[-1]
        # goal check when a node is chosen to be expanded 
        if node.val == 'g':
            print "PATH FOUND!!"
            print "Nodes explored: " + str(len(explored))
            return path
        # set gcost (IT'S JUST GOING TO BE THE PATH LENGTH!)
        gcost = len(path)
        # set position
        pos = node.pos
        # get all adjacent nodes, construct a new path and push it to the fringe queue
        for i in xrange(len(moves)): 
            if not (0 <= pos[0] + moves[i][0] < rows and 0 <= pos[1] + moves[i][1] < columns):
                continue
            neighbor = graph[pos[0] + moves[i][0]][pos[1] + moves[i][1]]
            if neighbor.val == 'o':
                continue
            if not neighbor in explored:
                # increment the gcost assuming the cost to travel one block = 1
                newgcost = 1 + gcost 
                # store the node / gcost pair in the dictionary
                explored[neighbor] = newgcost
                # calculate estimated distance from neighbor to goal
                hcost = dist(neighbor, goal)
                # calculate f cost by adding distance from goal to total path cost
                fcost = newgcost + hcost
                # copy the path by value and add the neighbor
                newPath = path[:]
                newPath.append(neighbor)
                fringe.put((newPath, fcost)) 


if results.bfs:
    # call BFS
    startTime = time.time()
    path = bfs(graph, start) 
    endTime = time.time()
    runtime = endTime - startTime
    print "BFS algorithm called"
    print "Runtime: " + str(runtime)
    # draw map with path, write path length in file
    length = "BFS: " + str(len(path)) + '\n\n' 
    mapString = writeMap(path)
    f.write(mapString)
    f.write(length)
    resetGraph(graph)


# Depth First Search
if results.dfs:
    # call DFS
    startTime = time.time()
    path = dfs(graph, start)
    endTime = time.time()
    runtime = endTime - startTime
    print "DFS algorithm called" 
    print "Runtime: " + str(runtime)
    # draw map with path, write path length in file
    length = "DFS: " + str(len(path)) + '\n\n'
    mapString = writeMap(path)
    f.write(mapString)
    f.write(length)
    resetGraph(graph)

# A* Search
if results.astar:
    # call A*
    startTime = time.time()
    path = astar(graph, start, goal)
    endTime = time.time()
    runtime = endTime - startTime
    print "A* algorithm called"
    print "Runtime: " + str(runtime)
    # draw map with path, write path length in file
    length = "A*: " + str(len(path)) + '\n\n'
    mapString = writeMap(path)
    f.write(mapString)
    f.write(length)
    resetGraph(graph)




