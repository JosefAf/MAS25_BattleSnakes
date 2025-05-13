# Le ULTIMATA BATTLE SNAKE AI - The Monte Carlo Tree Search

# 1 Helper classes for the Monte Carlo Tree Search
"""
Define Snake object using:
snake = Snake("ID", HP:100, BODY:[(5, 5), (5, 4), (5, 3)]) 

# Define food positions:
food = [(6, 5), (0, 0)] # List of tuples

# Create the game state using:
state = GameState(my_snake="ID", snakes=[snake, ...], food=food, width=11, height=11)

# Run a rollout using:
result = state.rollout() # Used in 2.2
"""


# 1.1  Class to represent a snake
class Snake:

    def __init__(self, id, body, health):
        self.id = id
        self.body = body
        self.head = body[0]
        self.health = health
        self.alive = True

    # Simulates movment of the snake in a given direction. Includes health reduction and death if health is 0, handles also edge case like eating food
    def move(self, direction, did_grow=False):
        dx, dy = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }[direction]

        new_head = (self.head[0] + dx, self.head[1] + dy)
        self.body.insert(0, new_head)

        if not did_grow:
            self.body.pop()

        self.health -= 1
        if self.health <= 0:
            self.alive = False

        if did_grow:
            self.health = 100


# 1.2 Class to represent the game state
class GameState:

    def __init__(self, my_snake, snakes, food, width, height, turn=0):
        self.my_snake = my_snake  # Store the id of our snake
        self.snakes = {
            s.id: s
            for s in snakes
        }  # Dict of Snake objects stored by id, to make it easier to access them.
        self.food = list(food)  # List of (x, y)
        self.width = width
        self.height = height
        self.turn = turn

    def copy(self):
        return GameState(self.my_snake, [
            Snake(s.id, list(s.body), s.health) for s in self.snakes.values()
        ], list(self.food), self.width, self.height, self.turn)

    # Return all snakes alive in given game state using their ids
    def get_alive_snakes(self):
        return {sid: s for sid, s in self.snakes.items() if s.alive}

    # Quickly check if game is over (TODO: maybe also handle teammate deaths?)
    def is_terminal(self):
        alive = self.get_alive_snakes()
        return self.my_snake not in alive or len(alive) <= 1

    # Get all safe moves for a given snake
    def get_safe_moves(self, snake: Snake):
        directions = ["up", "down", "left", "right"]
        safe = []

        for d in directions:
            dx, dy = {
                "up": (0, 1),
                "down": (0, -1),
                "left": (-1, 0),
                "right": (1, 0)
            }[d]
            new_head = (snake.head[0] + dx, snake.head[1] + dy)

            # Check if move is out of bounds
            if not (0 <= new_head[0] < self.width
                    and 0 <= new_head[1] < self.height):
                continue

            # Check if move would collide with another alive snake
            # TODO: as of now we don't handle head-to-head collisions with smaller snakes!
            if any(new_head in s.body for s in self.snakes.values()
                   if s.alive):
                continue

            safe.append(d)

        return safe or directions  # If no safe moves then return all

    # Simulate a turn using random move from given current state
    def simulate_turn(self):
        # Randomly choose moves for all alive snakes
        # TODO: maybe use some heuristic to choose moves for other snakes?
        planned_moves = {}
        for sid, snake in self.get_alive_snakes().items():
            if sid == self.my_snake.id:
                continue
            move = random.choice(self.get_safe_moves(snake))
            planned_moves[sid] = move

        # Execute planned moves

        eaten_food = set()  # Keep track of food that has been eaten

        for sid, move in planned_moves.items():
            snake = self.snakes[sid]
            new_head = self.calculate_new_head(snake.head, move)
            grow = new_head in self.food
            snake.move(move, did_grow=grow)
            if grow:
                eaten_food.add(new_head)
                self.food.remove(new_head)

        # Remove dead snakes from starvation or collision
        self.handle_deaths()

        self.turn += 1

    # Calculate new head position based on current head and move direction
    def calculate_new_head(self, head, move):
        dx, dy = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }[move]

        return (head[0] + dx, head[1] + dy)

    # Handle deaths of snakes due to wall collision, body collision, starvation, and head-to-head collision
    def handle_deaths(self):
        # Keep a occupied set of all snake body segments
        occupied = set()
        for sid, snake in self.get_alive_snakes().items():
            for segment in snake.body[1:]:
                occupied.add(segment)

        # Wall, body, starvation, and self collisions
        # TODO: Check for correct indexing for out of bounds
        for sid, snake in self.get_alive_snakes().items():
            # Handle wall collisions
            if not (0 <= snake.head[0] < self.width
                    and 0 <= snake.head[1] < self.height):
                snake.alive = False

            # Handle body collisions
            elif snake.head in occupied:
                snake.alive = False

            # Handle starvation
            elif snake.health <= 0:
                snake.alive = False

            # Handle self collisions
            elif snake.head in snake.body[1:]:
                snake.alive = False

        # Head-to-head case: i.e. multiple heads in the same cell
        head_map = {}
        for sid, snake in self.get_alive_snakes().items():
            head_map.setdefault(snake.head, []).append(
                sid)  # Group snakes by head position

        # Kill all snakes in the same cell, except the longest one
        for position, sids in head_map.items():
            if len(sids) > 1:
                # Find the longest snake or snakes
                lengths = [self.snakes[sid].length for sid in sids]
                max_length = max(lengths)
                longest_s = [
                    sid for sid in sids
                    if self.snakes[sid].length == max_length
                ]  # List of all snakes that have max lenght

                # Kill all snakes except the longest one in current cell
                if len(longest_s) == 1:
                    for sid in sids:
                        if sid != longest_s[0]:
                            self.snakes[sid].alive = False
                # Kill all snakes as they are the same length
                else:
                    for sid in sids:
                        self.snakes[sid].alive = False

    # Simulate rollout until terminal state or max steps reached
    def rollout(self, max_steps):
        step = 0
        while not self.is_terminal() and step < max_steps:
            self.simulate_turn()
            step += 1

        if self.my_snake not in self.get_alive_snakes():
            return "loss"

        elif len(
                self.get_alive_snakes()) == 1 and self.my_snake in self.snakes:
            return "win"

        return "draw"


# 2. Monte Carlo Tree Search Algorithm


# 2.1 Class to represent a node in the search tree
class MCTSNode:

    def __init__(self, state, parent=None, move=None):
        self.state = state  # Represented by a game state object from 1.2
        self.parent = parent  # Parent node in the search tree
        self.move = move  # The move that led to this state
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of times this node has led to a win

    # Check if all child moves are explored
    def is_fully_expanded(self):
        snake = self.state.snakes[self.state.my_snake]
        available_moves = self.state.get_safe_moves(snake)
        explored_moves = [child.move for child in self.children]
        return set(available_moves) == set(explored_moves)

    # Select the best child based on UCB1 formula
    def best_child(self, c=1.5):
        choices_weights = []
        for child in self.children:
            exploitation = child.wins / child.visits
            exploration = math.sqrt(2 * math.log(self.visits) / child.visits)
            choices_weights.append(exploitation + c * exploration)
        return self.children[choices_weights.index(max(choices_weights))]

    # Add a new child node with an unexplored move
    def expand(self):
        snake = self.state.snakes[self.state.my_snake]
        available_moves = self.state.get_safe_moves(snake)
        explored = [child.move for child in self.children]

        for move in available_moves:
            if move not in explored:
                # Simulate this move only for the root player
                new_state = self.state.copy()
                new_state.snakes[self.state.my_snake].move(move)
                new_state.handle_deaths()
                new_state.turn += 1
                child = MCTSNode(new_state, parent=self, move=move)
                self.children.append(child)
                return child


# 2.2 Main function to get best estimated move using MCTS
# TODO: put entire thing in a class


# Get the best move using Monte Carlo Tree Search
def get_best_move(root_state, time_limit=1):  # Time limit is in ms
    root = MCTSNode(state=root_state)
    end_time = time.time() + time_limit / 1000

    while time.time() < end_time:
        #print(time.time())
        node = tree_policy(root)
        result = rollout(
            node.state
        )  # Note: Why this complain? Why no work? :( Best guess something to do with expand bein none
        backpropagate(node, result)

    best = max(root.children, key=lambda c: c.visits)
    return best.move


# Tree policy: selection -> expansion
def tree_policy(node):
    while not node.state.is_terminal():
        if not node.is_fully_expanded():
            return node.expand()
        else:
            node = node.best_child()
    return node


# Peform a rollout from the given state
def rollout(state):
    rollout_state = state.copy()
    return rollout_state.rollout(max_steps=20)


# Backpropagate the result up the tree
# TODO: maybe add half wins for victory and possibly also for team snake wins?
def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        if result == "win":
            node.wins += 1
        node = node.parent
