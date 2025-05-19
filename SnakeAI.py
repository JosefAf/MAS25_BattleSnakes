import numpy as np
import random
import heapq
import math
from collections import deque


class SnakeAI:

    def __init__(self):
        self.prev_lengths = {}
        self.grid = None

    def reset(self):
        self.prev_lengths = {}

    def update_state(self, game_state):
        # --- load raw state ---
        self.board = game_state['board']
        self.my_snake_id = game_state['you']['id']
        self.width = self.board['width']
        self.height = self.board['height']
        self.food = {(f['x'], f['y']) for f in self.board['food']}
        self.snakes = {s['id']: s for s in self.board['snakes']}

        # --- detect “just ate” ---
        curr_lens = {sid: len(s['body']) for sid, s in self.snakes.items()}
        self.just_ate = {
            sid: (sid in self.prev_lengths
                  and curr_lens[sid] > self.prev_lengths[sid])
            for sid in self.snakes
        }
        self.prev_lengths = curr_lens

        # --- build the new probability grid ---
        self.build_probability_grid()

    # ──────────────────────────────────────────────────────────────────────────
    # 1) PROBABILITY GRID
    # ──────────────────────────────────────────────────────────────────────────
    def build_probability_grid(self, debug: bool = True):
        W, H = self.width, self.height
        grid = np.zeros((H, W), dtype=float)
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # a) block all body segments (and “just‐ate” tails) at 1.0
        for sid, s in self.snakes.items():
            body = s['body']
            xs = [pt['x'] for pt in body[:-1]]
            ys = [pt['y'] for pt in body[:-1]]
            grid[ys, xs] = 1.0
            tx, ty = body[-1]['x'], body[-1]['y']
            grid[ty, tx] = 1.0 if self.just_ate.get(sid, False) else grid[ty,
                                                                          tx]

        # stash base grid so expected_reach sees it
        self.grid = grid

        # b) for each *other* snake, distribute probability over its next‐move cells
        for sid, s in self.snakes.items():
            if sid == self.my_snake_id:
                continue

            hx, hy = s['body'][0]['x'], s['body'][0]['y']
            neck = tuple(s['body'][1].values())

            # collect legal next cells
            candidates = []
            for dx, dy in DIRS:
                nx, ny = hx + dx, hy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if (nx, ny) == neck:
                    continue
                if grid[ny, nx] >= 1.0:
                    continue
                candidates.append((nx, ny))

            if not candidates:
                continue

            # compute weights based on distance to food and flood-fill reachability
            weights = []
            for (nx, ny) in candidates:
                # distance to nearest food
                if self.food:
                    dists = [abs(nx - fx) + abs(ny - fy) for fx, fy in self.food]
                    min_dist = min(dists)
                else:
                    min_dist = W + H
                weight_dist = 1.0 / (min_dist + 1)

                # expected reachable area from this cell
                reach_area = self.expected_reach(nx, ny)

                weights.append(weight_dist * reach_area)

            total_weight = sum(weights)
            for (nx, ny), w in zip(candidates, weights):
                if total_weight > 0:
                    p = w / total_weight
                else:
                    p = 1.0 / len(candidates)
                grid[ny, nx] = max(grid[ny, nx], p)

        self.grid = grid
        if debug:
            for y in range(self.height - 1, -1, -1):
                row = self.grid[y]
                print(" ".join(f"{val:.2f}" for val in row))

    # ──────────────────────────────────────────────────────────────────────────
    # 2) “RISK‐AWARE” PATHFINDING
    # ──────────────────────────────────────────────────────────────────────────
    def expected_path_cost(self, start, goal, alpha=5.0):
        W, H = self.width, self.height
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_heap = [(h(start, goal), start)]
        g_score = {start: 0.0}

        while open_heap:
            f, current = heapq.heappop(open_heap)
            if current == goal:
                return g_score[current]

            for dx, dy in DIRS:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                p_block = self.grid[ny, nx]
                if p_block >= 1.0:
                    continue
                step_cost = 1.0 + alpha * p_block
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative_g
                    heapq.heappush(open_heap, (tentative_g + h(
                        (nx, ny), goal), (nx, ny)))

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # 3) PROBABILISTIC REACH (FLOOD)
    # ──────────────────────────────────────────────────────────────────────────
    def expected_reach(self, sx, sy):
        W, H = self.width, self.height
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        reach_p = np.zeros((H, W), dtype=float)
        reach_p[sy, sx] = 1.0

        q = deque([(sx, sy)])
        while q:
            x, y = q.popleft()
            base = reach_p[y, x]
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                p_block = self.grid[ny, nx]
                if p_block >= 1.0:
                    continue
                new_p = base * (1.0 - p_block)
                if new_p > reach_p[ny, nx]:
                    reach_p[ny, nx] = new_p
                    q.append((nx, ny))

        return reach_p.sum()

    # ──────────────────────────────────────────────────────────────────────────
    # 4) SAFE MOVES (avoid high-risk head-on collisions)
    # ──────────────────────────────────────────────────────────────────────────
    def get_Safe_Moves(self):
        head = self.snakes[self.my_snake_id]['body'][0]
        x, y = head['x'], head['y']
        DIRS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        # collect all moves not certain death (p_block < 1)
        moves = []
        for m, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                p = self.grid[ny, nx]
                if p < 1.0:
                    moves.append((m, p))

        if not moves:
            return []

        # if any move has zero risk, keep only those
        zero_moves = [m for m, p in moves if p == 0.0]
        if zero_moves:
            return zero_moves

        # otherwise keep only the move(s) with minimal risk
        min_p = min(p for _, p in moves)
        return [m for m, p in moves if p == min_p]

    def pick_best_move(self, scores):
        best = max(scores.values())
        choices = [m for m, s in scores.items() if s == best]
        print(f"Choices: {choices}")
        return random.choice(choices)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) SCORING FUNCTIONS USING THE PROBABILITY GRID
    # ──────────────────────────────────────────────────────────────────────────
    def get_Food_Score(self, safe_moves):
        if not self.food:
            return {m: 0.0 for m in safe_moves}

        head = self.snakes[self.my_snake_id]['body'][0]
        hx, hy = head['x'], head['y']

        # current best expected-cost to any food
        best0 = min((self.expected_path_cost((hx, hy), f) or np.inf)
                    for f in self.food)

        scores = {}
        for m in safe_moves:
            dx, dy = {
                'up': (0, 1),
                'down': (0, -1),
                'left': (-1, 0),
                'right': (1, 0)
            }[m]
            nx, ny = hx + dx, hy + dy

            if (nx, ny) in self.food:
                scores[m] = best0 + 10.0
            else:
                best1 = min((self.expected_path_cost((nx, ny), f) or np.inf)
                            for f in self.food)
                scores[m] = best0 - best1

        return scores

    def get_Space_Score(self, safe_moves):
        head = self.snakes[self.my_snake_id]['body'][0]
        hx, hy = head['x'], head['y']
        norm = 10.0 / (self.width * self.height)

        curr_space = self.expected_reach(hx, hy)
        scores = {}
        for m in safe_moves:
            dx, dy = {
                'up': (0, 1),
                'down': (0, -1),
                'left': (-1, 0),
                'right': (1, 0)
            }[m]
            nx, ny = hx + dx, hy + dy
            scores[m] = (self.expected_reach(nx, ny) - curr_space) * norm

        return scores

    def get_Enemy_Space_Score(self, safe_moves):
        me = self.snakes[self.my_snake_id]['body'][0]
        mx, my = me['x'], me['y']
        others = [
            s['body'][0] for sid, s in self.snakes.items()
            if sid != self.my_snake_id
        ]
        if not others:
            return {m: 0.0 for m in safe_moves}

        target = min(others, key=lambda h: abs(h['x'] - mx) + abs(h['y'] - my))
        ox, oy = target['x'], target['y']
        norm = 10.0 / (self.width * self.height)

        scores = {}
        for m in safe_moves:
            dx, dy = {
                'up': (0, 1),
                'down': (0, -1),
                'left': (-1, 0),
                'right': (1, 0)
            }[m]
            nx, ny = mx + dx, my + dy

            save = self.grid.copy()
            self.grid[ny, nx] = 1.0
            reach = self.expected_reach(ox, oy)
            self.grid = save

            scores[m] = -reach * norm

        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # 6) FINAL MOVE PICKER
    # ──────────────────────────────────────────────────────────────────────────
    def get_Next_Move(self, debug: bool = True):
        safe = self.get_Safe_Moves()
        if not safe:
            return random.choice(["up", "down", "left", "right"])

        me = self.snakes[self.my_snake_id]
        my_len = len(me['body'])
        health = me.get('health', 100)
        other_max = max(
            (len(s['body'])
             for sid, s in self.snakes.items() if sid != self.my_snake_id),
            default=0)
        consider_food = not (my_len >= other_max + 3 and health >= 25)

        fs = self.get_Food_Score(safe) if consider_food else {
            m: 0
            for m in safe
        }
        for move, score in fs.items(): # for if no move is found
            if math.isnan(score):
                fs[move] = 0

        
        ss = self.get_Space_Score(safe)
        es = self.get_Enemy_Space_Score(safe)

        wf = 1.0 if consider_food else 0.0
        ws, we = 4.0, 2.2

        total = {m: wf * fs[m] + ws * ss[m] + we * es[m] for m in safe}
        if debug:
            print("Food:", fs, "\nSpace:", ss, "\nEnemy:", es, "\nTotal:",
                  total)

        return self.pick_best_move(total)
