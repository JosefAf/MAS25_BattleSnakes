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
        hazards (set of (x, y)): Hazards (not yet used).
        snakes (dict): Mapping snake_id -> {'health': int, 'body': list of (x,y)}
        you_id (str): Our snake's identifier.
    """
    def __init__(self, board: dict = None, you_id: str = None):
        # Allow __new__ construction without args
        if board is None:
            return
        # Store board dimensions and data
        self.width = board['width']
        self.height = board['height']
        self.food = {(f['x'], f['y']) for f in board['food']}
        self.hazards = {(h['x'], h['y']) for h in board.get('hazards', [])}
        # Parse all snakes
        self.snakes = {}
        for s in board['snakes']:
            sid = s['id']
            self.snakes[sid] = {
                'health': s['health'],
                'body': [(seg['x'], seg['y']) for seg in s['body']]
            }
        self.you_id = you_id

    @staticmethod
    def move_position(pos, direction):
        """Move pos by direction vector and return new tuple."""
        dx, dy = DIRECTIONS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def in_bounds(self, pos):
        """Return True if pos is within board limits."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, pos):
        """Return True if any snake occupies pos."""
        return any(pos in snake['body'] for snake in self.snakes.values())

    def get_legal_moves(self, snake_id):
        """Return list of safe moves (no wall or body collisions)."""
        head = self.snakes[snake_id]['body'][0]
        safe = []
        for move in DIRECTIONS:
            new_head = self.move_position(head, move)
            if not self.in_bounds(new_head):
                continue
            if self.is_occupied(new_head):
                continue
            safe.append(move)
        return safe or list(DIRECTIONS.keys())

    def apply_moves(self, moves: dict):
        """
        Apply given moves for some snakes; others move randomly.
        Returns a new GameState.
        """
        new_snakes = deepcopy(self.snakes)
        new_food = set(self.food)
        # Determine new heads
        new_heads = {}
        for sid, snake in new_snakes.items():
            direction = moves.get(sid, random.choice(self.get_legal_moves(sid)))
            new_heads[sid] = self.move_position(snake['body'][0], direction)
        # Update snakes
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
        # Resolve deaths
        survivors = {}
        for sid, snake in new_snakes.items():
            head = snake['body'][0]
            if not self.in_bounds(head) or snake['health'] <= 0:
                continue
            if any(head in other['body'] for oid, other in new_snakes.items() if oid != sid):
                continue
            survivors[sid] = snake
        # Build new state
        new_state = GameState.__new__(GameState)
        new_state.width = self.width
        new_state.height = self.height
        new_state.food = new_food
        new_state.hazards = self.hazards
        new_state.snakes = survivors
        new_state.you_id = self.you_id
        return new_state

    def is_terminal(self):
        """Return True if game ended (our snake dead or only one snake)."""
        return (self.you_id not in self.snakes) or (len(self.snakes) <= 1)

    def get_result(self):
        """Return game outcome: 1.0 win, 0.0 loss, else heuristic [0,1]."""
        if self.you_id not in self.snakes:
            return 0.0
        if len(self.snakes) == 1:
            return 1.0
        my = self.snakes[self.you_id]
        my_score = my['health'] + 2 * len(my['body'])
        other_scores = [s['health'] + 2 * len(s['body'])
                        for sid, s in self.snakes.items() if sid != self.you_id]
        best_other = max(other_scores)
        return (my_score - best_other + 100) / 200

class MCTSNode:
    """Node in MCTS, storing stats, children, and depth."""
    def __init__(self, state: GameState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []           # list of (action, MCTSNode)
        self.visits = 0
        self.wins = 0.0
        # Depth from root (root depth=0)
        self.depth = parent.depth + 1 if parent else 0
        # Untried actions for our snake, or empty if terminal
        if state.is_terminal() or state.you_id not in state.snakes:
            self.untried_actions = []
        else:
            self.untried_actions = state.get_legal_moves(state.you_id)

    def expand(self):
        """Expand one untried action and return new child node."""
        action = self.untried_actions.pop()
        next_state = self.state.apply_moves({self.state.you_id: action})
        child = MCTSNode(next_state, parent=self)
        self.children.append((action, child))
        return child

    def best_child(self, c_param=1.41):
        """Return child node with highest UCB1 score."""
        best_score = -float('inf')
        best = None
        for action, child in self.children:
            exploit = child.wins / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def rollout(self, max_rollout=20):
        """Simulate random play to depth limit and return result."""
        current = self.state
        for _ in range(max_rollout):
            if current.is_terminal():
                break
            moves = {sid: random.choice(current.get_legal_moves(sid))
                     for sid in current.snakes}
            current = current.apply_moves(moves)
        return current.get_result()

    def backpropagate(self, result):
        """Propagate result up to the root."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def subtree_max_depth(self):
        """Compute max depth in this subtree (including this node)."""
        if not self.children:
            return self.depth
        return max(child.subtree_max_depth() for _, child in self.children)


def mcts_search(root_state: GameState, time_limit=0.4):
    """
    Run MCTS to select best move.
    Prints iteration count and best subtree depth.
    Returns chosen action.
    """
    root = MCTSNode(root_state)
    iterations = 0
    end_time = time.time() + time_limit
    while time.time() < end_time:
        node = root
        # 1. Selection
        while not node.untried_actions and node.children:
            node = node.best_child()
        # 2. Expansion
        if node.untried_actions:
            node = node.expand()
        # 3. Simulation
        result = node.rollout()
        # 4. Backpropagation
        node.backpropagate(result)
        iterations += 1
    # 5. Choose best action by visits
    if root.children:
        best_action, best_node = max(root.children, key=lambda item: item[1].visits)
        max_subtree_depth = best_node.subtree_max_depth()
        print(f"MCTS iterations: {iterations}, best subtree max depth: {max_subtree_depth}")
        return best_action
    # Fallback if no children (e.g., immediately terminal): pick random legal move
    fallback = random.choice(root_state.get_legal_moves(root_state.you_id))
    print(f"MCTS iterations: {iterations}, no expansion—fallback move: {fallback}")
    return fallback
