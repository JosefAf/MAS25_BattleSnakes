# For more info see docs.battlesnake.com

import random
import typing
import math
from collections import deque  #For double-ended queue in flood fill algorithm
import heapq  #For priority queue in A* algorithm

# Directions and their vector offsets
DIRECTIONS = {
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0),
}


class BaseSnakeAI:

    def __init__(self, game_state):
        self.game_state = game_state
        self.you = game_state["you"]
        self.board = game_state["board"]
        #self.game = game_state["game"]
        self.occupied_cells = self.setOccupiedCells()

    def setOccupiedCells(self):
        """Method takes different kinds of occupied cells (border positions & snake bodies' 
        positions) and combines them in a list of occupied cells. The positions are tuples 
        of x and y coordinates."""
        occupied_cells = []
        #Add borders to occupied cells
        for x in range(self.board["width"]):  #range(k) means 0 to k-1
            occupied_cells.append((x, -1))  #Adding bottom border
        for x in range(self.board["width"]):
            occupied_cells.append(
                (x, self.board["height"]))  #Adding top border
        for y in range(self.board["height"]):
            occupied_cells.append((-1, y))  #Adding left border
        for y in range(self.board["height"]):
            occupied_cells.append(
                (self.board["width"], y))  #Adding right border

        #Add snake bodies' positions at current turn, to occupied cells
        for snake in self.board["snakes"]:
            for body_part in snake[
                    "body"]:  #body_part is a dict with keys "x" and "y"
                occupied_cells.append((body_part["x"], body_part["y"]))

        return occupied_cells

    def getReachableCells(self, head_pos):
        """Method returns set of reachable cells from current snake position, using flood 
        fill algorithm with breadth-first search. Note that reachable cells can be reached, 
        but may be unsafe to move into, e.g. if they lead to a dead end."""

        my_head = self.you["body"][0]
        #Using a double-ended queue instead of a list, for faster .add and .pop operations
        #(indexing is faster with lists tho)
        queue = deque((my_head["x"], my_head["y"]))  #Adding cells as tuples
        visited_cells = set()
        #Using hashset for faster lookup among set of visited cells
        #(checking if a cell has been visited or not)

        while queue:  #run loop while queue is not empty
            #pop first element from queue, O(1) complexity :D
            x, y = queue.popleft()

            #for each possible movement direction:
            """USE GET NEIGHTBORDS METHOD INSTEAD OF THIS TODO TODO TODO TODOT ERKEDODKWAOP"""
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy  #nx, ny are coords of neighboring cells

                if not (0 <= nx < self.board['width']
                        and 0 <= ny < self.board['height']):
                    continue  # neighboring cell is out of bounds, skip it
                if (nx, ny) in self.occupied_cells or (nx,
                                                       ny) in visited_cells:
                    continue  # neighboring cell is occupied or already visited, skip it

                visited_cells.add(
                    (nx, ny))  #Adding item to hashset is on average O(1)
                queue.append((nx, ny))  # Appending to deque is O(1) :D

        return visited_cells

    def getNeighbors(self, pos):
        width = self.board["width"]
        height = self.board["height"]
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for nx, ny in candidates
                if 0 <= nx < width and 0 <= ny < height]


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#FFC0CB",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    print("MOVE")

    my_head = game_state["you"]["body"][0]  # Coordinates of your head as dict
    my_head_pos = (my_head["x"], my_head["y"])  # Coords of head as tuple
    my_snake_AI = BaseSnakeAI(game_state)

    #print(my_snake_AI.occupied_cells)

    potential_moves = [
    ]  #List of moves the head of the snake can make without dying
    for direction in DIRECTIONS:
        new_pos = (my_head_pos[0] + DIRECTIONS[direction][0],
                   my_head_pos[1] + DIRECTIONS[direction][1])
        if new_pos in my_snake_AI.occupied_cells:
            #print("Direction to occupied cell: " + direction)
            continue
        else:
            potential_moves.append(direction)

    #potential_new_pos is now a list of 0 to 3 tuples of new positions for the head to take without dying
    #If no potential moves, i.e len(potential_new_pos) == 0, then snake is trapped and will die,
    # so return "up" to go to heaven
    if len(potential_moves) == 0:
        next_move = "up"
        return {"move": next_move}

    #Reaching this point means there are nonzero potential moves
    #Let's evaluate each potential move and choose the best one according to the following heuristic:
    # Assuming the current state of the game is static into next turn, the best move is the one that
    # leads us closer to food while remaining in an as open area as possible (i.e. not leading to a dead end)
    # FOR EACH MOVE - calculate manhattan distance to nearest food
    #               - use flood fill to count the number of free reachable cells from new position
    # BONUS POINTS ARE AWARDED FOR - moving into position that has food in it
    # NOTE: A good move should have a low overall score, not a high one!

    #Calculate manhattan distance to nearest food if there is food on the board, otherwise set it to 0
    if len(game_state["board"]["food"]) > 0:
        closest_food_pos = getClosestFood(my_head_pos,
                                          game_state["board"]["food"])
        #Calculate score for each potential move and return the move with the lowest score
        next_move = calculateMoveScore(my_head_pos, potential_moves,
                                       closest_food_pos, game_state["board"])
    else:
        closest_food_pos = 0
        #Calculate score for each potential move and return the move with the lowest score
        next_move = calculateMoveScore(my_head_pos, potential_moves,
                                       closest_food_pos, game_state["board"])

    #print(closest_food_pos)

    #ADD HEURISTIC PART FOR FLOOD FILL HERE

    return {"move": next_move}


def calculateMoveScore(my_head_pos, potential_moves, closest_food_pos, board):
    """Calculate score for each potential move and return the move with the lowest score,
    according to the heuristic described in the move() function."""

    move_scores = {}  #Dict of moves and their scores

    for direction in potential_moves:
        #Calculate manhattan distance to nearest food
        potential_pos = (my_head_pos[0] + DIRECTIONS[direction][0],
                         my_head_pos[1] + DIRECTIONS[direction][1])
        distance_to_closest_food = manhattanDistance(potential_pos,
                                                     closest_food_pos)
        #Use flood fill to count the number of free reachable cells from new position TODO TODO TODO TODO
        move_scores[
            direction] = distance_to_closest_food * 5  #Add directions as keys and their scores as values

    #LINE BELOW WOULD BE NICE IF IT WORKED MY DAWG
    #next_move = max(move_scores.keys(), key = move_scores.get) #Get key with lowest value in dict

    #Get move with lowest score
    next_move = "up"
    for key in move_scores:
        if move_scores[key] == min(move_scores.values()):
            next_move = key

    return next_move


def getClosestFood(start_pos, food_dict):
    min_distance = math.inf
    closest_food_pos = 0

    for food in food_dict:
        food_pos = (food["x"], food["y"])
        result = manhattanDistance(start_pos, food_pos)
        if result < min_distance:
            min_distance = result
            closest_food_pos = food_pos

    return closest_food_pos


def manhattanDistance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
