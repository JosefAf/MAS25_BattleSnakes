import numpy as np
import random
from collections import deque
from copy import deepcopy

"""
NOTES:
- We need to debug the get obstacle grid, as there are some bugs [Done]
- Maybe we can use a score for the grid, instead of 0 and 1, instead of blocking we score a cell based on how much we want to avoid it
- Fine tune the scores
- Add score for reducing other snakes reachable space
- Dynamic score, only add food score if we are low on hp or not larger than other snakes
"""

class SnakeAI:

    def __init__(self):
        self.prev_lengths = {} # remember the length of each snake before the move
        self.grid = None

    def update_state(self, game_state):
        """
        get new game state and update the fields.
        Also compute self.just_ate by comparing self.prev_lengths
        Then rebuild your obstacle grid
        """
        self.board       = game_state['board']
        self.my_snake_id = game_state['you']['id']
        self.width       = self.board['width']
        self.height      = self.board['height']
        self.food        = {(f['x'], f['y']) for f in self.board['food']}
        self.snakes      = {s['id']: s for s in self.board['snakes']}

        # to know if a snake just ate, we compare the length of the snake before and after the move
        # therefore the snakeAI has its internal fields updated instead of reinstanced. This is why we need to update the prev_lengths field.
        current_lengths     = {sid: len(s['body']) for sid, s in self.snakes.items()}
        self.just_ate       = {
            sid: (sid in self.prev_lengths and current_lengths[sid] > self.prev_lengths[sid])
            for sid in self.snakes
        }
        self.prev_lengths   = current_lengths

        # get obstacle grid
        self.get_Obstacle_Grid(True)
    
    def reset(self):
        # clear the lenght dict for a new game
        self.prev_lengths = {}

    
    def get_Obstacle_Grid(self, debug: bool = False):
        """
        - Create a 2D array representing the game board.
        - Mark all snake body segments excluding the tail as blocked.
        - Edge case for tail, if snake has just ate, then the tail is blocked, otherwise it is free.
        - The adjacent cells to other snakes are also blocked.
        - Edge case if our snake is 1 cell away from other snake, we need to check if that other snake is longer than us, if so mark it adjacent cell that also is adjacent to us as blocked. Otherwise if it is shorter than us, that adjacent cell to both of us is free.
        """
        # Important notice: here neighbor/adjacent cells are the cells that are 1 manhattan distance away from a current cell. Ie. reachable in 1 move.
        
        W, H = self.width, self.height
        gm = self.get_Manhattan
        DIRS = [(0,1),(0,-1),(1,0),(-1,0)]

        grid = np.zeros((H, W), dtype=np.uint8)

        # 1) Block body segments for all snakes (except tails)
        for sid, s in self.snakes.items():
            body = s['body']
            # a) block all but the tail
            xs = [pt['x'] for pt in body[:-1]]
            ys = [pt['y'] for pt in body[:-1]]
            grid[ys, xs] = 1

            # b) tail: blocked only if that snake just grew
            tx, ty = body[-1]['x'], body[-1]['y']
            grid[ty, tx] = 1 if self.just_ate.get(sid, False) else 0

        # 2) get our head neighbors for head to head logic
        me = self.snakes[self.my_snake_id]
        mx, my = me['body'][0]['x'], me['body'][0]['y']
        my_nec = (me['body'][1]['x'], me['body'][1]['y']) # exclude my neck
        my_len = len(me['body'])
        my_nei = {
            (mx+dx, my+dy)
            for dx, dy in DIRS
            if 0 <= mx+dx < W and 0 <= my+dy < H and (mx+dx, my+dy) != my_nec
        } # represents all cells our snake can move to

        # 3) Block neighbors of other heads + handle head to head conflicts
        for sid, s in self.snakes.items():
            if sid == self.my_snake_id:
                continue
            ox, oy = s['body'][0]['x'], s['body'][0]['y']
            other_nec = (s['body'][1]['x'], s['body'][1]['y']) # exclude other neck
            other_len = len(s['body'])
            other_nei = set()
            # we want to block all cells other snakes can move to
            for dx, dy in DIRS:
                nx, ny = ox+dx, oy+dy
                if 0 <= nx < W and 0 <= ny < H and (ox+dx, oy+dy) != other_nec:
                    grid[ny, nx] = 1
                    other_nei.add((nx, ny)) 

            # if we’re longer, consider shared* neighbor cells as free
            if my_len > other_len:
                for (x, y) in my_nei & other_nei:
                    grid[y, x] = 0

        self.grid = grid
        # debug print
        if debug:
            for row in grid[::-1]:
                print(' '.join(str(int(v)) for v in row))

    def get_Manhattan(self, pos1, pos2):
        """
        Get Manhattan distance between two points.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_Safe_Moves(self, debug: bool = False):
        """
        Based from the grid, get all possible moves that are not blocked.
        """
        W, H = self.width, self.height
        grid = self.grid
        head = self.snakes[self.my_snake_id]['body'][0]
        x, y = head['x'], head['y']

        # Possible moves
        DIRS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
        }

        safe = []
        for move, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            # check in-bounds
            if 0 <= nx < W and 0 <= ny < H:
                # append all moves that are not blocked in the grid
                if grid[ny, nx] == 0:
                    safe.append(move)

        if debug:
            print(f"Safe moves: {safe}")
        
        return safe

    def pick_best_move(self, scores):
        """
        Given a dict of move and score, return the one with the highest score.
        """

        # the max score in the scores dict
        max_score = max(scores.values())

        # get all moves that have the best score (could be more)
        best_moves = [move for move, sc in scores.items() if sc == max_score]

        # if more than one move have high scores then pick at random
        return random.choice(best_moves)

    def get_Next_Move(self, debug: bool = False):

        # Get all safe moves
        safe_moves = self.get_Safe_Moves(False)
        if not safe_moves:
            return "up"
        
        # Score each move based on different criteria
        food_Scores = self.get_Food_Score(safe_moves)
        space_Scores = self.get_Space_Score(safe_moves)
        enemy_Space_Scores = self.get_Enemy_Space_Score(safe_moves)
        
        # total score for each move
        wf = 1  # weight of food score
        ws = 4  # weight of space score
        we = 2.2  # weight for enemy space score
        
        total_Scores  = {m: wf * food_Scores[m] + ws * space_Scores[m] + we * enemy_Space_Scores[m] for m in safe_moves}
        
        next_move = self.pick_best_move(total_Scores)

        if debug:
            food_debug = {m: wf * food_Scores[m] for m in safe_moves}
            print("Food Scores:", food_debug)
            space_debug = {m: ws * space_Scores[m] for m in safe_moves}
            print("Space Scores:", space_debug)
            enemy_debug = {m: we * enemy_Space_Scores[m] for m in safe_moves}
            print("Enemy Space Scores:", enemy_debug)

        return next_move

    def get_Food_Score(self, safe_moves):
        """
        Given the safe moves we want to score each move to a “food score” that reflects how much closer it brings us to the nearest food. Higher is better.
        """
        DIRS = {
            'up': (0,  1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1,  0),
        }
        
        # our head position
        head = self.snakes[self.my_snake_id]['body'][0]
        hx, hy = head['x'], head['y']
        
        if not self.food:
            # no food on board so all moves have zero score
            return {m: 0 for m in safe_moves}

        # closest food distance to our head
        curr_min = min(self.get_Manhattan((hx, hy), f) for f in self.food)
        
        scores = {}
        for move in safe_moves:
            dx, dy = DIRS[move]
            nx, ny = hx + dx, hy + dy
        
            # if we’d land on food, give a big bonus
            if (nx, ny) in self.food:
                scores[move] = curr_min + 10  # we want to always eat food if possible
                continue
        
            # otherwise measure new closest food distance
            new_min = min(self.get_Manhattan((nx, ny), f) for f in self.food)
            scores[move] = curr_min - new_min
        
        return scores

    def get_Space_Score(self, safe_moves):
        """
        Given a list of safe moves, run a flood fill from current head and from each candidate head, and score each move as: score_space(move) = (reachable_after_move - reachable_now) * (MAX_FOOD_BONUS / (width * height))
    
        Thus all cells would be the same as +MAX_FOOD_BONUS, and losing cells gives you a negative penalty of the same scale.
        """
        
        DIRS = {
            'up': (0,1), 
            'down': (0,-1), 
            'left': (-1,0), 
            'right': (1,0)
        }
    
        W, H = self.width, self.height
        head = self.snakes[self.my_snake_id]['body'][0]
        hx, hy = head['x'], head['y']
    
        # weight so that a full board gain = food landing bonus to make the scales match
        MAX_FOOD_BONUS = 10.0
        norm = MAX_FOOD_BONUS / (W * H) # represent how much we value space
    
        # current reachable cells
        curr_reach = self.flood(hx, hy)
    
        # evaluate each candidate
        scores = {}
        for move in safe_moves:
            dx, dy = DIRS[move]
            nx, ny = hx + dx, hy + dy
            reach = self.flood(nx, ny)
            delta = reach - curr_reach
            scores[move] = delta * norm
    
        return scores

    def get_Enemy_Space_Score(self, safe_moves):
        """
        Score each move by how much it *reduces* the closest enemy's reachable space.
        Higher score means the enemy has fewer reachable cells after our move.
        """
        DIRS = {
            'up': ( 0,  1),
            'down': ( 0, -1),
            'left': (-1,  0),
            'right': ( 1,  0),
        }
        
        # our head position
        me = self.snakes[self.my_snake_id]['body'][0]
        mx, my = me['x'], me['y']

        # get other heads
        others = [
            s['body'][0]
            for sid, s in self.snakes.items()
            if sid != self.my_snake_id
        ]
        
        if not others:
            # no other snakes so all moves have zero score
            return {m: 0 for m in safe_moves}

        # find closest by Manhattan
        target = min(others, key=lambda h: self.get_Manhattan((mx, my), (h['x'], h['y'])))
        ox, oy = target['x'], target['y'] # closest enemy head

        scores = {}
        for move in safe_moves:
            dx, dy = DIRS[move]
            nx, ny = mx + dx, my + dy

            # copy grid and block our new head
            grid_copy = self.grid.copy()
            grid_copy[ny, nx] = 1

            # temp swap in grid_copy for flood
            # maybe make it so flood fill also take grid as an argument
            orig_grid = self.grid
            self.grid = grid_copy
            
            reachable = self.flood(ox, oy)

            # DON'T FORGET: restore original grid
            self.grid = orig_grid

            # score = negative reachable (so fewer = better)
            scores[move] = -reachable

        return scores
    
    # flood fill using BFS returning count of reachable cells from (x,y)
    def flood(self, x, y):
        W, H = self.width, self.height
        seen = {(x, y)}
        q = deque([(x, y)])
        count = 0
        while q:
            x, y = q.popleft()
            count += 1
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x+dx, y+dy
                if (0 <= nx < W and 0 <= ny < H
                    and self.grid[ny, nx] == 0
                    and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return count

