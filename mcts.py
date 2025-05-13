import random
import time
import math
from copy import deepcopy

# Directions and their vector offsets
DIRECTIONS = {
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0),
}

class GameState:
    """
    Represents a snapshot of the Battlesnake board.

    Attributes:
        width (int): Board width.
        height (int): Board height.
        food (set of (x, y)): Locations of food.
        hazards (set of (x, y)): Locations of hazards (not used yet).
        snakes (dict): Mapping from snake_id to {
            'health': int,
            'body': list of (x, y)
        }
        you_id (str): ID of our snake.
    """
    def __init__(self, board: dict = None, you_id: str = None):
        # Initialize from existing attributes if called via __new__
        if board is None:
            return

        # Board dimensions
        self.width = board['width']
        self.height = board['height']

        # Food positions as a set for O(1) lookup
        self.food = {(f['x'], f['y']) for f in board['food']}

        # Hazards (currently not used)
        self.hazards = {(h['x'], h['y']) for h in board.get('hazards', [])}

        # Parse snakes into a convenient dict
        self.snakes = {}
        for s in board['snakes']:
            snake_id = s['id']
            health = s['health']
            body = [(seg['x'], seg['y']) for seg in s['body']]
            self.snakes[snake_id] = {'health': health, 'body': body}

        # Our snake's ID
        self.you_id = you_id

    @staticmethod
    def move_position(pos, direction):
        """Return a new position tuple after moving in the given direction."""
        dx, dy = DIRECTIONS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def in_bounds(self, pos):
        """Check if a position is within the board boundaries."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, pos):
        """Check if a position is occupied by any snake body segment."""
        for snake in self.snakes.values():
            if pos in snake['body']:
                return True
        return False

    def get_legal_moves(self, snake_id):
        """
        Get a list of legal (safe) moves for the specified snake ID.
        A move is legal if it does not hit a wall or any snake's body.
        """
        head = self.snakes[snake_id]['body'][0]
        legal = []
        for move in DIRECTIONS:
            new_head = self.move_position(head, move)
            if not self.in_bounds(new_head):
                continue
            if self.is_occupied(new_head):
                continue
            legal.append(move)
        return legal or list(DIRECTIONS.keys())

    def apply_moves(self, moves: dict):
        """
        Apply a move for each snake and return the resulting GameState.
        moves: {snake_id: direction}, others move randomly.
        """
        new_snakes = deepcopy(self.snakes)
        new_food = set(self.food)

        # Compute new head positions
        new_heads = {}
        for sid, snake in new_snakes.items():
            direction = moves.get(sid, random.choice(self.get_legal_moves(sid)))
            new_heads[sid] = self.move_position(snake['body'][0], direction)

        # Update each snake's body and health
        for sid, snake in new_snakes.items():
            head = new_heads[sid]
            ate = head in new_food
            if ate:
                snake['body'] = [head] + snake['body']
                snake['health'] = 100
                new_food.remove(head)
            else:
                snake['body'] = [head] + snake['body'][:-1]
                snake['health'] -= 1

        # Resolve collisions and deaths
        survivors = {}
        for sid, snake in new_snakes.items():
            head = snake['body'][0]
            if not self.in_bounds(head) or snake['health'] <= 0:
                continue
            if any(head in other['body'] for oid, other in new_snakes.items() if oid != sid):
                continue
            survivors[sid] = snake

        # Build new GameState via __new__ to skip init
        new_state = GameState.__new__(GameState)
        new_state.width = self.width
        new_state.height = self.height
        new_state.food = new_food
        new_state.hazards = self.hazards
        new_state.snakes = survivors
        new_state.you_id = self.you_id
        return new_state

    def is_terminal(self):
        """Return True if game over (our snake dead or only one remains)."""
        return (self.you_id not in self.snakes) or (len(self.snakes) <= 1)

    def get_result(self):
        """
        Return 1.0 for win, 0.0 for loss, or heuristic [0,1] otherwise.
        """
        if self.you_id not in self.snakes:
            return 0.0
        if len(self.snakes) == 1:
            return 1.0

        my = self.snakes[self.you_id]
        my_score = my['health'] + 2 * len(my['body'])
        other_scores = [s['health'] + 2 * len(s['body']) for sid,s in self.snakes.items() if sid != self.you_id]
        best_other = max(other_scores)
        return (my_score - best_other + 100) / 200


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    """
    def __init__(self, state: GameState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        # If terminal or our snake is dead, no actions to try
        if state.is_terminal() or state.you_id not in state.snakes:
            self.untried_actions = []
        else:
            self.untried_actions = state.get_legal_moves(state.you_id)

    def expand(self):
        """Create and return a new child node for one untried action."""
        action = self.untried_actions.pop()
        next_state = self.state.apply_moves({self.state.you_id: action})
        child = MCTSNode(next_state, parent=self)
        self.children.append((action, child))
        return child

    def best_child(self, c_param=1.41):
        """Select child with highest UCB1 value."""
        best_score = -float('inf')
        best_child = None
        for action, child in self.children:
            exploit = child.wins / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def rollout(self, max_rollout=20):
        """Simulate random playout up to max_rollout turns."""
        current = self.state
        for _ in range(max_rollout):
            if current.is_terminal():
                break
            moves = {sid: random.choice(current.get_legal_moves(sid)) for sid in current.snakes}
            current = current.apply_moves(moves)
        return current.get_result()

    def backpropagate(self, result):
        """Update this node and its ancestors with result."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


def mcts_search(root_state: GameState, time_limit=0.4):
    """
    Perform MCTS from root_state within time_limit seconds.
    Return the best action for our snake.
    """
    root = MCTSNode(root_state)
    end = time.time() + time_limit
    while time.time() < end:
        node = root
        # Selection
        while not node.untried_actions and node.children:
            node = node.best_child()
        # Expansion
        if node.untried_actions:
            node = node.expand()
        # Simulation
        result = node.rollout()
        # Backpropagation
        node.backpropagate(result)

    # Pick action with highest visit count
    best = max(root.children, key=lambda item: item[1].visits)
    return best[0]