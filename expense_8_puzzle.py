import sys  
from queue import Queue
from datetime import datetime  
import datetime
import heapq

def is_goal_state(state, goal_state):
    return state == goal_state


def find_blank_tile(start_state):
    for i, row in enumerate(start_state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j     
#bfs algorithm  

def bfs_alg(start_state, goal_state, save_trace=False):
    q = Queue()
    visited = set()
    moves = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1
    depth = 0
    
    if save_trace:
        # Create datetime.txt file and open it for writing
        filename = "datetime.txt"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as file:                              
            file.write(f"Search trace ({now})\n\n")
    
    q.put((start_state, [], 0))
    
    while not q.empty():
        current, path, cost = q.get()
        node_popped += 1
        visited.add(str(current))
        
        if current == goal_state:
            if save_trace:
                with open(filename, "a") as file:
                    file.write("Nodes Popped: {}\n".format(node_popped))
                    file.write("Nodes Expanded: {}\n".format(node_expanded))
                    file.write("Nodes Generated: {}\n".format(node_generated))
                    file.write("Max Fringe Size: {}\n".format(max_fringe_size))
                    file.write("Solution Found at depth {} with cost of {}\n".format(depth, cost))
                    file.write("Steps:\n{}\n".format('\n'.join(path)))
            return True
        
        row, col = find_blank_tile(current)
        for action, (drow, dcol) in moves.items():
            new_row, new_col = row+drow, col+dcol
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in current]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                if str(new_state) not in visited:
                    node_generated += 1
                    cost_to_move = new_state[row][col] 
                    path_copy = path.copy()
                    path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                    q.put((new_state, path_copy, cost+cost_to_move))
        
        node_expanded += 1
        if q.qsize() > max_fringe_size:
            max_fringe_size = q.qsize()
        if cost > depth:
            depth = cost
            
        if save_trace:
            # Write fringe and closed set contents to the file
            with open(filename, "a") as file:
                file.write("Iteration {}\n".format(node_expanded))
                file.write("Closed: {}\n".format(str(visited)))
                file.write("Fringe: [\n")
                for item in list(q.queue):
                    file.write("\t{}\n".format(str(item)))
                file.write("]\n\n")
            
    return False

#ucs algorithm

def ucs_alg(start_state, goal_state, save_trace=False):
    queue = [(0, start_state, [], 0)]
    visited = set()
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1
    depth = 0

    while queue:
        _, current, path, cost = heapq.heappop(queue)
        node_popped += 1
        visited.add(str(current))

        if current == goal_state:
            if save_trace:
                with open("datetime.txt", "a") as file:
                    file.write("Nodes Popped: {}\n".format(node_popped))
                    file.write("Nodes Expanded: {}\n".format(node_expanded))
                    file.write("Nodes Generated: {}\n".format(node_generated))
                    file.write("Max Fringe Size: {}\n".format(max_fringe_size))
                    file.write("Solution Found at depth {} with cost of {}\n".format(depth, cost))
                    file.write("Steps:\n{}\n".format('\n'.join(path)))
            else:
                print("Nodes Popped:", node_popped)
                print("Nodes Expanded:", node_expanded)
                print("Nodes Generated:", node_generated)
                print("Max Fringe Size:", max_fringe_size)
                print("Solution Found at depth", depth, "with cost of", cost)
                print("Steps:\n", '\n'.join(path))
            return True

        row, col = find_blank_tile(current)
        for action, (drow, dcol) in moves.items():
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in current]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                if str(new_state) not in visited:
                    node_generated += 1
                    cost_to_move = new_state[row][col]
                    path_copy = path.copy()
                    path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                    heapq.heappush(queue, (cost + cost_to_move, new_state, path_copy, cost + cost_to_move))

        node_expanded += 1
        if len(queue) > max_fringe_size:
            max_fringe_size = len(queue)
        if cost > depth:
            depth = cost

    if save_trace:
        with open("datetime.txt", "a") as file:
            file.write("Closed: {}\n".format(str(visited)))
            file.write("Fringe: [\n")
            for item in queue:
                file.write("\t{}\n".format(str(item)))
            file.write("]\n\n")

    return False

#Heuristic function

def heuristic(start_state, goal_state):
    start_state=[num for row in start_state for num in row]
    goal_state=[num for row in goal_state for num in row]
    distance = 0
    for i in range(9):
        if start_state[i] != goal_state[i]:
            (goal_i, goal_j) = divmod(goal_state.index(start_state[i]), 3)
            (i, j) = divmod(i, 3)
            distance += abs(i - goal_i) + abs(j - goal_j)
    return distance
#a star algorithm
def a_star_alg(start_state, goal_state, save_trace=False):
    q = []
    visited = set()
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1
    depth = 0
    cost = 0
    priority = cost + heuristic(start_state, goal_state)
    heapq.heappush(q, (priority, start_state, [], cost))

    while q:
        _, current, path, cost = heapq.heappop(q)
        node_popped += 1
        visited.add(str(current))

        if current == goal_state:
            if save_trace:
                with open("datetime.txt", "a") as file:
                    file.write("Nodes Popped: {}\n".format(node_popped))
                    file.write("Nodes Expanded: {}\n".format(node_expanded))
                    file.write("Nodes Generated: {}\n".format(node_generated))
                    file.write("Max Fringe Size: {}\n".format(max_fringe_size))
                    file.write("Solution Found at depth {} with cost of {}\n".format(depth, cost))
                    file.write("Steps:\n{}\n".format('\n'.join(path)))
            else:
                print("Nodes Popped:", node_popped)
                print("Nodes Expanded:", node_expanded)
                print("Nodes Generated:", node_generated)
                print("Max Fringe Size:", max_fringe_size)
                print("Solution Found at depth", depth, "with cost of", cost)
                print("Steps:\n", '\n'.join(path))
            return True

        row, col = find_blank_tile(current)
        for action, (drow, dcol) in moves.items():
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in current]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                if str(new_state) not in visited:
                    node_generated += 1
                    cost_to_move = new_state[row][col]
                    path_copy = path.copy()
                    path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                    new_cost = cost + cost_to_move
                    new_priority = new_cost + heuristic(new_state, goal_state)
                    heapq.heappush(q, (new_priority, new_state, path_copy, new_cost))

        node_expanded += 1
        if len(q) > max_fringe_size:
            max_fringe_size = len(q)
        if cost > depth:
            depth = cost

    if save_trace:
        with open("datetime.txt", "a") as file:
            file.write("Closed: {}\n".format(str(visited)))
            file.write("Fringe: [\n")
            for item in q:
                file.write("\t{}\n".format(str(item)))
            file.write("]\n\n")

    return False

#dfs algorithm
def dfs_alg(start_state, goal_state, save_trace=False):
    stack = [(start_state, [], 0)]
    visited = set()
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1
    depth = 0

    if save_trace:
        filename = "datetime.txt"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as file:
            file.write(f"Search trace ({now})\n\n")

    while stack:
        current, path, cost = stack.pop()
        node_popped += 1
        visited.add(str(current))

        if current == goal_state:
            if save_trace:
                with open(filename, "a") as file:
                    file.write("Nodes Popped: {}\n".format(node_popped))
                    file.write("Nodes Expanded: {}\n".format(node_expanded))
                    file.write("Nodes Generated: {}\n".format(node_generated))
                    file.write("Max Fringe Size: {}\n".format(max_fringe_size))
                    file.write("Solution Found at depth {} with cost of {}\n".format(depth, cost))
                    file.write("Steps:\n{}\n".format('\n'.join(path)))
            else:
                print("Nodes Popped:", node_popped)
                print("Nodes Expanded:", node_expanded)
                print("Nodes Generated:", node_generated)
                print("Max Fringe Size:", max_fringe_size)
                print("Solution Found at depth", depth, "with cost of", cost)
                print("Steps:\n", '\n'.join(path))
            return True

        row, col = find_blank_tile(current)
        for action, (drow, dcol) in moves.items():
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in current]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                if str(new_state) not in visited:
                    node_generated += 1
                    cost_to_move = new_state[row][col]
                    path_copy = path.copy()
                    path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                    stack.append((new_state, path_copy, cost + cost_to_move))

        node_expanded += 1
        if len(stack) > max_fringe_size:
            max_fringe_size = len(stack)
        if cost > depth:
            depth = cost

        if save_trace:
            with open(filename, "a") as file:
                file.write("Closed: {}\n".format(str(visited)))
                file.write("Fringe: [\n")
                for item in stack:
                    file.write("\t{}\n".format(str(item)))
                file.write("]\n\n")

    return False


#greedy algoriithm
def greedy_alg(start_state, goal_state, save_trace=False):
    q = []
    visited = set()
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1
    depth = 0
    cost = 0

    priority = heuristic(start_state, goal_state)
    heapq.heappush(q, (priority, start_state, [], cost))

    if save_trace:
        filename = "datetime.txt"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as file:
            file.write(f"Search trace ({now})\n\n")

    while q:
        _, current, path, cost = heapq.heappop(q)
        node_popped += 1
        visited.add(str(current))

        if current == goal_state:
            if save_trace:
                with open(filename, "a") as file:
                    file.write("Nodes Popped: {}\n".format(node_popped))
                    file.write("Nodes Expanded: {}\n".format(node_expanded))
                    file.write("Nodes Generated: {}\n".format(node_generated))
                    file.write("Max Fringe Size: {}\n".format(max_fringe_size))
                    file.write("Solution Found at depth {} with cost of {}\n".format(depth, cost))
                    file.write("Steps:\n{}\n".format('\n'.join(path)))
            else:
                print("Nodes Popped:", node_popped)
                print("Nodes Expanded:", node_expanded)
                print("Nodes Generated:", node_generated)
                print("Max Fringe Size:", max_fringe_size)
                print("Solution Found at depth", depth, "with cost of", cost)
                print("Steps:\n", '\n'.join(path))
            return True

        row, col = find_blank_tile(current)
        for action, (drow, dcol) in moves.items():
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in current]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                if str(new_state) not in visited:
                    node_generated += 1
                    cost_to_move = new_state[row][col]
                    path_copy = path.copy()
                    path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                    new_cost = cost + cost_to_move
                    new_priority = heuristic(new_state, goal_state)
                    heapq.heappush(q, (new_priority, new_state, path_copy, new_cost))

        node_expanded += 1
        if len(q) > max_fringe_size:
            max_fringe_size = len(q)
        if cost > depth:
            depth = cost

        if save_trace:
            with open(filename, "a") as file:
                file.write("Closed: {}\n".format(str(visited)))
                file.write("Fringe: [\n")
                for item in q:
                    file.write("\t{}\n".format(str(item)))
                file.write("]\n\n")

    return False


#dls algorithm
def dls_alg(start_state, goal_state, max_lim=4, save_trace=False):
    stack = [(start_state, [], 0)]
    visited = set()
    moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1

    if save_trace:
        filename = "datetime.txt"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "w") as file:
            file.write(f"Search trace ({now})\n\n")

    while stack:
        current, path, depth = stack.pop()
        node_popped += 1
        visited.add(str(current))

        if current == goal_state:
            if save_trace:
                with open(filename, "a") as file:
                    file.write("Nodes Popped: {}\n".format(node_popped))
                    file.write("Nodes Expanded: {}\n".format(node_expanded))
                    file.write("Nodes Generated: {}\n".format(node_generated))
                    file.write("Max Fringe Size: {}\n".format(max_fringe_size))
                    file.write("Solution Found at depth {} with cost of {}\n".format(depth, depth))  # depth used as cost
                    file.write("Steps:\n{}\n".format('\n'.join(path)))
            else:
                print("Nodes Popped:", node_popped)
                print("Nodes Expanded:", node_expanded)
                print("Nodes Generated:", node_generated)
                print("Max Fringe Size:", max_fringe_size)
                print("Solution Found at depth", depth)
                print("Steps:\n", '\n'.join(path))
            return True

        if depth < max_lim:
            row, col = find_blank_tile(current)
            for action, (drow, dcol) in moves.items():
                new_row, new_col = row + drow, col + dcol
                if 0 <= new_row < 3 and 0 <= new_col < 3:
                    new_state = [row[:] for row in current]
                    new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                    if str(new_state) not in visited:
                        node_generated += 1
                        path_copy = path.copy()
                        path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                        stack.append((new_state, path_copy, depth + 1))

        node_expanded += 1
        if len(stack) > max_fringe_size:
            max_fringe_size = len(stack)

        if save_trace:
            with open(filename, "a") as file:
                file.write("Closed: {}\n".format(str(visited)))
                file.write("Fringe: [\n")
                for item in stack:
                    file.write("\t{}\n".format(str(item)))
                file.write("]\n\n")

    return False


#ids algorithm 

def ids_alg(start_state, goal_state):
    depth = 0
    while True:
        print("Checking depth limited search with max depth", depth)
        if dls_alg(start_state, goal_state, depth):
            return True
        depth += 1

#ids algorithm for datetime.txt 

def ids_alg_dt(start_state, goal_state):
    visited = set()
    moves = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
    node_popped = 0
    node_expanded = 0
    node_generated = 1
    max_fringe_size = 1
    depth_lim = 0
    
    while depth_lim <= 31:
        q = []
        heapq.heappush(q, (depth_lim, start_state, []))
        
        while q:
            _, current, path = heapq.heappop(q)
            node_popped += 1
            visited.add(str(current))
            
            if is_goal_state(current, goal_state):
                return path, node_popped, node_expanded, node_generated, max_fringe_size, depth_lim
            
            if len(path) < depth_lim:
                row, col = find_blank_tile(current)
                for action, (drow, dcol) in moves.items():
                    new_row, new_col = row+drow, col+dcol
                    if 0 <= new_row < 3 and 0 <= new_col < 3:
                        new_state = [row[:] for row in current]
                        new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                        if str(new_state) not in visited:
                            node_generated += 1
                            path_copy = path.copy()
                            path_copy.append("Move {} {} ".format(current[new_row][new_col], action.upper()))
                            heapq.heappush(q, (len(path_copy), new_state, path_copy))
            node_expanded += 1
            if len(q) > max_fringe_size:
                max_fringe_size = len(q)
        depth_lim+= 1
        
    return None, node_popped, node_expanded, node_generated, max_fringe_size, depth_lim


def main():
    if len(sys.argv) < 4:
        print("Command Line argument should be: python3 expense_8_puzzle.py start.txt goal.txt Method flag")
    else:
        arg1 = sys.argv[1] if len(sys.argv) > 1 else 'start.txt'
        arg2 = sys.argv[2] if len(sys.argv) > 2 else 'goal.txt'
        arg3 = sys.argv[3] if len(sys.argv) > 3 else 'a_star'
        arg4 = sys.argv[4] if len(sys.argv) > 4 else 'False'

        # Opening text file
        with open(arg1, 'r') as file:
            file_contents = file.read()
        rows = file_contents.split('\n')
        start_state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            numbers = rows[i].split(' ')
            for j in range(3):
                start_state[i][j] = int(numbers[j])

        with open(arg2, 'r') as file:
            file_contents = file.read()
        rows = file_contents.split('\n')
        goal_state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            numbers = rows[i].split(' ')
            for j in range(3):
                goal_state[i][j] = int(numbers[j])

        save_trace = True if arg4.lower() == 'true' else False

        if arg3 == 'ids':
            if save_trace:
                ids_alg_dt(start_state, goal_state)
            else:
                ids_alg(start_state, goal_state)
        elif arg3 == 'bfs':
            if save_trace:
                bfs_alg(start_state, goal_state, save_trace=True)
            else:
                bfs_alg(start_state, goal_state)
        elif arg3 == 'dfs':
            if save_trace:
                dfs_alg(start_state, goal_state, save_trace=True)
            else:
                dfs_alg(start_state, goal_state)
        elif arg3 == 'ucs':
            if save_trace:
                ucs_alg(start_state, goal_state, save_trace=True)
            else:
                ucs_alg(start_state, goal_state)
        elif arg3 == 'dls':
            if save_trace:
                dls_alg(start_state, goal_state, save_trace=True)
            else:
                dls_alg(start_state, goal_state)
        elif arg3 == 'greedy':
            if save_trace:
                greedy_alg(start_state, goal_state, save_trace=True)
            else:
                greedy_alg(start_state, goal_state)
        elif arg3 == 'a_star':
            if save_trace:
                a_star_alg(start_state, goal_state, save_trace=True)
            else:
                a_star_alg(start_state, goal_state)

if __name__ == '__main__':
    main()
