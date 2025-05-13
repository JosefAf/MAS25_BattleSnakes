from collections import deque #For double-ended queue in flood fill algorithm
import heapq #For priority queue in A* algorithm

class BaseSnakeAI:

    def __init__(self, game_state):
        self.game_state = game_state
        self.you = game_state["you"]
        self.board = game_state["board"]
        #self.game = game_state["game"]
        self.occupied_cells = self.setOccupiedCells()

    
    def setOccupiedCells(self):
        """Method takes different kinds of occupied cells (border positions & snake bodies' positions) 
        and combines them in a list of occupied cells. The positions are tuples of x and y coordinates."""
        occupied_cells = []
        #Add borders to occupied cells
        for x in range(self.board["width"]): #range(k) means 0 to k-1
            occupied_cells.append((x, 0)) #Adding bottom border
            occupied_cells.append((x, self.board["height"]-1)) #Adding top border
        for y in range(self.board["height"]):
            occupied_cells.append((0, y)) #Adding left border
            occupied_cells.append((self.board["width"]-1, y)) #Adding right border

        #Add snake bodies' positions at current turn, to occupied cells
        for snake in self.board["snakes"]:
            for body_part in snake["body"]: #body_part is a dict with keys "x" and "y"
                occupied_cells.append((body_part["x"], body_part["y"]))
        
        return occupied_cells

    
    def getReachableCells(self):
        """Method returns set of reachable cells from current snake position, using flood fill algorithm with breadth-first search.             Note that reachable cells can be reached, but may be unsafe to move into, e.g. if they lead to a dead end."""
    
        my_head = self.you["body"][0]
        #Using a double-ended queue instead of a list, for faster .add and .pop operations (indexing is faster with lists tho)
        queue = deque((my_head["x"], my_head["y"])) #Adding cells as tuples, because using dicts is cumbersome for meeeeeeeee
        visited_cells = set() #Using hashset for faster lookup among set of visited cells (checking if a cell has been visited or not)

        while queue: #run loop while queue is not empty
            x, y = queue.popleft() #pop first element from queue, O(1) complexity :D
            
            #for each possible movement direction:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy #nx, ny are coords of neighboring cells
                
                if not (0 <= nx < self.board['width'] and 0 <= ny < self.board['height']):
                    continue # neighboring cell is out of bounds, skip it
                if (nx, ny) in self.occupied_cells or (nx, ny) in visited_cells:
                    continue # neighboring cell is occupied or already visited, skip it
                    
                visited_cells.add((nx, ny)) #Adding item to hashset is on average O(1)
                queue.append((nx, ny))  # Appending to deque is O(1) :D
      
        return visited_cells

    
    def findPath(self, start_pos, goal_pos):
        """Method finds path from tuple start_pos to tuple goal_pos using A* algorithm. Returns path as list of tuples, 
        where each tuple is a cell's coordinates. Returns empty list if no path is found. Has the following helper-funcs:
        getNeighbors(), reconstructPath(), manhattanDistance()."""
        
        open_set = [] #List representing open set/ priority queue
        heapq.heappush(open_set, (0, start_pos)) #Push start position to open set with priority 0
        came_from = {} # For reconstructing path
        g_score = {start_pos: 0} # Cost from start to current node
        
        while open_set:
            _, current_pos = heapq.heappop(open_set) #_ is used to ignore the priority value (first given output)
        
            if current_pos == goal_pos:
                return self.reconstructPath(came_from, current_pos)
        
            for neighbor in self.getNeighbors(current_pos):
                if neighbor in self.occupied_cells:
                    continue # Skip occupied cells
        
                tentative_g = g_score[current_pos] + 1  # Assuming uniform movement cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current_pos
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattanDistance(neighbor, goal_pos)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None  # No path found
    
    def getNeighbors(self, pos):
        width = self.board["width"]
        height = self.board["height"]
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for nx, ny in candidates if 0 <= nx < width and 0 <= ny < height]
    
    def reconstructPath(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current] #This gives the parent cell of current
            path.append(current) #Add parent cell to path
        path.reverse() #Path is built from goal to start, so we need to reverse it
        return path
    
    def manhattanDistance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])