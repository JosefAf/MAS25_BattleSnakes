import random
import time
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
    def __init__(self, board: dict, you_id: str):
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
        # If no safe moves, return all moves to force a decision
        return legal or list(DIRECTIONS.keys())

    def apply_moves(self, moves: dict):
        """
        Apply a move for each snake and return the resulting GameState.
        moves: {snake_id: direction}, others move randomly if absent.
        """
        # Deep copy current snake data
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
                # Grow: new head + full body
                snake['body'] = [head] + snake['body']
                snake['health'] = 100  # reset health
                new_food.remove(head)
            else:
                # Move: new head + drop tail
                snake['body'] = [head] + snake['body'][:-1]
                snake['health'] -= 1

        # Resolve collisions and wall/health deaths
        survivors = {}
        for sid, snake in new_snakes.items():
            head = snake['body'][0]
            # Died by wall or starvation
            if not self.in_bounds(head) or snake['health'] <= 0:
                continue
            # Died by collision with other body
            collision = any(
                (head in other['body'])
                for oid, other in new_snakes.items() if oid != sid
            )
            if collision:
                continue
            survivors[sid] = snake

        # Build new state without calling initializer
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
        if self.you_id not in self.snakes:
            return True
        return len(self.snakes) <= 1

    def get_result(self):
        """
        Return 1.0 for win, 0.0 for loss, or heuristic [0,1] otherwise.
        Heuristic = (my_score - best_other_score + 100) / 200
        """
        # Loss if we're gone
        if self.you_id not in self.snakes:
            return 0.0
        # Win if sole survivor
        if len(self.snakes) == 1:
            return 1.0

        # Heuristic comparison
        my = self.snakes[self.you_id]
        my_score = my['health'] + 2 * len(my['body'])
        other_scores = [s['health'] + 2 * len(s['body'])
                        for sid,s in self.snakes.items() if sid != self.you_id]
        best_other = max(other_scores)
        return (my_score - best_other + 100) / 200


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    """
    def __init__(self, state: GameState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []  # list of child MCTSNode
        self.visits = 0
        self.wins = 0.0
        # Actions not yet tried from this state for our snake
        self.untried_actions = state.get_legal_moves(state.you_id)

    def expand(self):
        """Create and return a new child node for one untried action."""
        action = self.untried_actions.pop()
        # Apply our action, opponents random
        next_state = self.state.apply_moves({self.state.you_id: action})
        child = MCTSNode(next_state, parent=self)
        self.children.append((action, child))
        return child

    def best_child(self, c_param=1.41):
        """Select child with highest UCB1 value."""
        choices_weights = []
        for action, child in self.children:
            # UCB1: (wins/visits) + c * sqrt(ln(N)/visits)
            exploitation = child.wins / child.visits
            exploration = c_param * ( (2 * math.log(self.visits) / child.visits) ** 0.5 )
            choices_weights.append(exploitation + exploration)
        # Return child with max weight
        return self.children[choices_weights.index(max(choices_weights))][1]

    def rollout(self, max_rollout=20):
        """Simulate a random playout from current state up to max_rollout."""
        current_state = self.state
        for _ in range(max_rollout):
            if current_state.is_terminal():
                break
            # Random moves for all snakes
            moves = {
                sid: random.choice(current_state.get_legal_moves(sid))
                for sid in current_state.snakes
            }
            current_state = current_state.apply_moves(moves)
        return current_state.get_result()

    def backpropagate(self, result):
        """Update this node and ancestors with rollout result."""
        self.visits += 1
        self.wins += result
        if self.parent:
            # The reward for parent is from its perspective: invert for two-player?
            self.parent.backpropagate(result)


def mcts_search(root_state: GameState, time_limit=0.4):
    """
    Perform MCTS from root_state within time_limit seconds.
    Return the best action for our snake.
    """
    root = MCTSNode(root_state)
    end_time = time.time() + time_limit
    while time.time() < end_time:
        node = root
        # 1. Selection
        while node.untried_actions == [] and node.children:
            node = node.best_child()
        # 2. Expansion
        if node.untried_actions:
            node = node.expand()
        # 3. Simulation
        result = node.rollout()
        # 4. Backpropagation
        node.backpropagate(result)

    # Choose the move with most visits
    best_move, best_node = max(
        root.children,
        key=lambda item: item[1].visits
    )
    return best_move