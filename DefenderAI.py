import numpy as np
import random
from collections import deque
from copy import deepcopy
import heapq
"""Differences AttackerAI and DefenderAI:
- AttackerAI uses a probability grid instead of a binary grid."""


class DefenderAI:

    def __init__(self):
        self.prev_lengths = {
        }  # remember the length of each snake before the move
        self.grid = None

    def update_state(self, game_state, attacker_data):
        """
        get new game state and update the fields.
        Also compute self.just_ate by comparing self.prev_lengths
        Then rebuild your obstacle grid
        """
        self.board = game_state['board']
        self.my_snake_id = game_state['you']['id']
        self.width = self.board['width']
        self.height = self.board['height']
        self.food = {(f['x'], f['y'])
                     for f in self.board['food']
                     }  #save food positions as tuples in a hashset
        self.snakes = {
            s['id']: s
            for s in self.board['snakes']
        }  #save snake-dicts as dict with snake id as key to each snake-dict.

        # to know if a snake just ate, we compare the length of the snake before and after the move
        # therefore the snakeAI has its internal fields updated instead of reinstanced. This is why we need to update the prev_lengths field.
        current_lengths = {
            sid: len(s['body'])
            for sid, s in self.snakes.items()
        }
        self.just_ate = {
            sid: (sid in self.prev_lengths
                  and current_lengths[sid] > self.prev_lengths[sid])
            for sid in self.snakes
        }
        self.prev_lengths = current_lengths

        # get obstacle grid
        self.get_Obstacle_Grid(attacker_data, True)  #True parameter gives debug print of grid

    def reset(self):
        # clear the length dict for a new game
        self.prev_lengths = {}

    def get_Obstacle_Grid(self, attacker_data, debug: bool = True):
        """
        - Create a 2D array representing the game board.
        - Mark all snake body segments excluding the tail as blocked.
        - Edge case for tail, if snake has just ate, then the tail is blocked, otherwise it is free.
        - The adjacent cells to other snakes are also blocked.
        - Edge case if our snake is 1 cell away from other snake, we need to check if that other snake is longer than us, if so mark adjacent cell that also is adjacent to us as blocked. Otherwise if it is shorter than us, that adjacent cell to both of us is free.
        """
        # Important notice: here neighbor/adjacent cells are the cells that are 1 manhattan distance away from a current cell. Ie. reachable in 1 move.

        W, H = self.width, self.height
        gm = self.get_Manhattan  #UNUSED
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  #right, left, up, down
        DIRECTIONS = {"up": (1, 0), "down": (-1, 0), "left": (0, -1), "right":(0, 1)}

        grid = np.zeros(
            (H, W), dtype=np.uint8
        )  #storing the type numpy.uint8 - unsigned 8-bit integer (0 to 255)
        
        # 1) Block body segments for all snakes (except tails unless they just ate)
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
        mx, my = me['body'][0]['x'], me['body'][0]['y']  #my head position
        my_nec = (me['body'][1]['x'], me['body'][1]['y'])  # exclude my neck
        my_len = len(me['body'])
        my_nei = {
            (mx + dx, my + dy)
            for dx, dy in DIRS
            if 0 <= mx + dx < W and 0 <= my + dy < H and (mx + dx,
                                                          my + dy) != my_nec
        }
        #my_nei represents all cells our snake can move to

        # 3) Block neighbors of other heads + handle head to head conflicts
        for sid, s in self.snakes.items():
            if sid == self.my_snake_id:
                continue
            ox, oy = s['body'][0]['x'], s['body'][0][
                'y']  #other snake's head position
            other_nec = (s['body'][1]['x'], s['body'][1]['y']
                         )  # exclude other neck
            other_len = len(s['body'])  #other snake length
            other_nei = set()  #other snake's neighbors
            # we want to block all cells other snakes can move to
            # first we get neighbors of other snake's head and set them as blocked in grid
            for dx, dy in DIRS:
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < W and 0 <= ny < H and (ox + dx,
                                                    oy + dy) != other_nec:
                    grid[ny,
                        nx] = 1  #assign value 1 (blocked) to this position in grid
                    other_nei.add((nx, ny))

            # if we’re longer, consider shared neighbor cells as free, unless it is in position of our or enemy body part that is not a head (because colliding there would kill us).
            if my_len > other_len:
                my_body_except_head = set((pt['x'], pt['y']) for pt in me['body'][1:])
                other_body_except_head = set((pt['x'], pt['y']) for pt in s['body'][1:])
                for (x, y) in my_nei & other_nei:
                    if (x, y) not in my_body_except_head or other_body_except_head:
                        grid[y, x] = 0

        # 4) If we have attacker data, block attacker's next move cell for the defender:
        if attacker_data is not None:
            living_snakes = self.board["snakes"]
            for snake in living_snakes:
                if snake["id"] == attacker_data["ID"]:
                    attacker_head = snake["body"][0]
                    attacker_head_pos = (attacker_head["x"], attacker_head["y"])
                    attacker_next_move = attacker_data["next_move"]
                    attacker_next_pos = (attacker_head_pos[0] + DIRECTIONS[attacker_next_move][0], 
                                         attacker_head_pos[1] + DIRECTIONS[attacker_next_move][1])
                    if 0 <= attacker_next_pos[0] < W and 0 <= attacker_next_pos[1] < H:
                        grid[attacker_next_pos[1], attacker_next_pos[0]] = 1
        
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

    def get_Neighbors(self, pos):
        """Returns list of neighboring cells of a given cell position (tuple). Filters out cells that are out of bounds. MAKE SURE TO FILTER OUT SNAKE BODY POSITIONS FROM LIST RETURNED BY THIS METHOD IF NECESSARY!"""
        width = self.board["width"]
        height = self.board["height"]
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in candidates
                if 0 <= nx < width and 0 <= ny < height]

    def get_Dijkstra(self, start_pos, goal_pos):
        """Uses A* to get the Dijkstra distance (shortest distance while taking obstacles into account) between two cell positions the board. Returns None if no path is found. Returns length of path if path is found (int)."""

        open_set = [
        ]  #List representing open set/ priority queue. Used as heapq (min-heap, lowest prio-value first)
        heapq.heappush(
            open_set,
            (0, start_pos))  #Push start position to open set with priority 0
        came_from = {
        }  #For reconstructing path, this is a dict of child cell (key) - parent cell (value) pairs
        g_scores = {
            start_pos: 0
        }  #Cost from start to current node, cost to come per pos in dict

        while open_set:
            _, current_pos = heapq.heappop(
                open_set
            )  #_ is used to ignore the priority value (first given output)

            if current_pos == goal_pos:
                #Construct path from start to goal, then compute its length
                path = [goal_pos]
                while current_pos in came_from:  #True when current_pos is key in came_from dict
                    current_pos = came_from[
                        current_pos]  #This gives the parent cell of current
                    path.append(current_pos)  #Add parent cell to path
                #If we'd want to use this path, we'd need to reverse it first
                #path.reverse() #Path is built from goal to start, so we need to reverse it
                return len(
                    path
                ) - 1  #Distance from start to goal is len(path)-1 since path contains both start pos and goal pos

            #Get neighbor cells to current_pos and filter out all those that are occupied by snake bodies or borders.
            for neighbor in self.get_Neighbors(current_pos):
                if self.grid[neighbor[1], neighbor[0]] == 1:
                    continue  # Skip occupied cells

                preliminary_g_score = g_scores[
                    current_pos] + 1  #preliminary, because we might've found a better g_score before
                #if neighbor not visited before or if we now have found a shorter path than previously, to same neighbor:
                if neighbor not in g_scores or preliminary_g_score < g_scores[
                        neighbor]:
                    came_from[neighbor] = current_pos
                    g_scores[neighbor] = preliminary_g_score
                    f_score = preliminary_g_score + self.get_Manhattan(
                        neighbor, goal_pos)
                    heapq.heappush(open_set, (f_score, neighbor))
                #else we do nothing, because we've already found a shorter path to this neighbor before

        return None  # No path found

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

    def get_Next_Move(self, debug: bool = True):
        all_moves = ["up", "left", "down", "right"]

        # Get all safe moves
        safe_moves = self.get_Safe_Moves(False)
        # If no safe moves, return "up" to go to heaven
        if not safe_moves:
            random.choice(all_moves)

        # maybe put this in its own function
        me = self.snakes[self.my_snake_id]
        my_len = len(me['body'])
        health = me["health"]

        # lengths of all other snakes
        other_lengths = [
            len(s['body']) for sid, s in self.snakes.items()
            if sid != self.my_snake_id
        ]

        max_other_len = max(other_lengths) if other_lengths else 0

        # only skip food if we are >=2 longer than everyone AND health >= 25
        consider_food = not (my_len >= max_other_len + 2 and health >= 25)

        if consider_food:
            food_Scores = self.get_Food_Score(safe_moves)
        else:
            food_Scores = {m: 0 for m in safe_moves}

        # Score each move based on different criteria
        #food_Scores = self.get_Food_Score(safe_moves)
        space_Scores = self.get_Space_Score(safe_moves)
        enemy_Space_Scores = self.get_Enemy_Space_Score(safe_moves)

        # total score for each move
        wf = 1  # weight of food score
        ws = 4  # weight of space score
        we = 2.2  # weight for enemy space score

        total_Scores = {
            m: wf * food_Scores[m] + ws * space_Scores[m] +
            we * enemy_Space_Scores[m]
            for m in safe_moves
        }

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

        if not self.food:
            # no food on board so all moves have zero score
            return {m: 0 for m in safe_moves}

        DIRS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
        }

        # our head position
        head = self.snakes[self.my_snake_id]['body'][0]
        hx, hy = head['x'], head['y']

        # closest food distance to our head
        curr_min = np.inf
        for f in self.food:
            distance = self.get_Dijkstra((hx, hy), f)
            if distance is not None and distance < curr_min:
                curr_min = distance

        if curr_min == np.inf:
            # no path found to any food, all moves have zero score
            return {m: 0 for m in safe_moves}

        scores = {}
        for move in safe_moves:
            dx, dy = DIRS[move]
            nx, ny = hx + dx, hy + dy

            # if we’d land on food, give a big bonus
            if (nx, ny) in self.food:
                scores[
                    move] = curr_min #+ 10  # we want to always eat food if possible
                continue

            # otherwise measure new closest food distance
            new_curr_min = np.inf
            for f in self.food:
                distance = self.get_Dijkstra((nx, ny), f)
                if distance is not None and distance < new_curr_min:
                    new_curr_min = distance

            if new_curr_min == np.inf:
                # no path found to any food for this move
                scores[move] = 0

            else:
                scores[move] = curr_min - new_curr_min

        return scores

    def get_Space_Score(self, safe_moves):
        """
        Given a list of safe moves, run a flood fill from current head and from each candidate head, and score each move as: score_space(move) = (reachable_after_move - reachable_now) * (MAX_FOOD_BONUS / (width * height))

        Thus all cells would be the same as +MAX_FOOD_BONUS, and losing cells gives you a negative penalty of the same scale.
        """

        DIRS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        W, H = self.width, self.height
        head = self.snakes[self.my_snake_id]['body'][0]
        hx, hy = head['x'], head['y']

        # weight so that a full board gain = food landing bonus to make the scales match
        MAX_FOOD_BONUS = 11
        norm = MAX_FOOD_BONUS / (W * H)  # represent how much we value space

        # current reachable cells
        #curr_reach = self.flood(hx, hy)

        # evaluate each candidate
        scores = {}
        for move in safe_moves:
            dx, dy = DIRS[move]
            nx, ny = hx + dx, hy + dy
            reach = self.flood(nx, ny)
            #delta = reach - curr_reach
            #scores[move] = delta * norm
            scores[move] = reach * norm

        return scores

    def get_Enemy_Space_Score(self, safe_moves):
        """
        Score each move by how much it *reduces* the closest enemy's reachable space.
        Higher score means the enemy has fewer reachable cells after our move.
        """
        DIRS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
        }

        # weighting
        W, H = self.width, self.height
        MAX_FOOD_BONUS = 11
        norm = MAX_FOOD_BONUS / (W * H)

        # our head position
        me = self.snakes[self.my_snake_id]['body'][0]
        mx, my = me['x'], me['y']

        # get other heads
        others = [
            s['body'][0] for sid, s in self.snakes.items()
            if sid != self.my_snake_id
        ]

        if not others:
            # no other snakes so all moves have zero score
            return {m: 0 for m in safe_moves}

        # find closest by Manhattan
        target = min(others,
                     key=lambda h: self.get_Manhattan((mx, my),
                                                      (h['x'], h['y'])))
        ox, oy = target['x'], target['y']  # closest enemy head

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

            reach = self.flood(ox, oy)

            # DON'T FORGET: restore original grid
            self.grid = orig_grid

            # score = negative reachable (so fewer = better)
            scores[move] = -reach * norm

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
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < W and 0 <= ny < H and self.grid[ny, nx] == 0
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return count
