Overview
This repository contains the implementation of the modified 8-puzzle problem (Expense 8 puzzle problem) using various search algorithms.

Description
Your task is to build an agent to solve a modified version of the 8-puzzle problem, where the number on the tile also represents the cost of moving that tile. The goal is to determine the order of moves required to reach a desired configuration.

Objectives
Implement the following search algorithms:
Breadth First Search (BFS)
Uniform Cost Search (UCS)
Depth First Search (DFS)
Depth Limited Search (DLS)
Iterative Deepening Search (IDS)
Greedy Search
A* Search (default if no method is given)
Evaluate the search algorithms on the modified 8-puzzle problem.
Dump search trace if specified.
Command Line Invocation
expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>
<start-file>: Path to the start configuration file.
<goal-file>: Path to the goal configuration file.
<method>: Search algorithm to use (bfs, ucs, dfs, dls, ids, greedy, a*).
<dump-flag>: If true, search trace is dumped for analysis.
Example Usage
expense_8_puzzle.py start.txt goal.txt a* true
Sample Output
Nodes Popped: 97
Nodes Expanded: 64
Nodes Generated: 173
Max Fringe Size: 77
Solution Found at depth 12 with cost of 63.
Steps:
    Move 7 Left
    Move 5 Up
    Move 8 Right
    Move 7 Down
    Move 5 Left
    Move 6 Down
    Move 3 Right
    Move 2 Right
    Move 1 Up
    Move 4 Up
    Move 7 Left
    Move 8 Left

