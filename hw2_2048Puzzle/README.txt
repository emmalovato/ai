Emma Peterson
edp2117
edp2117@columbia.edu

This program has a player AI and a computer AI, both of which just call the alphabeta function to get their next move. The player calls the function as the Maximizing player (according to the minimax algorithm), and the computer calls the function as the Minimizing player. 

The alphabeta function used an evaluation function that combines a few different heuristics to calculate the total score of a given board state. It starts with the sum of each board tile, and adds or subtracts points based on the position of the maximum tile. Points are added if the tile is in a corner or on the edge of the board, and they are subtracted if the tile is in the center of the board. The difference between each tile and its neighbors is also subtracted from the total score. The score is increased if there are many free tiles on the board, and the score is decreased if there are few. 