import numpy as np
import random
import heapq
import math
from collections import deque
import copy

class SnakeAI:
    def __init__(self):
        # track previous lengths for growth detection
        self.prev_lengths = {}
        # probability grid of opponent moves
        self.grid = None

    def reset(self):
        # clear history between games
        self.prev_lengths = {}

    def update_state(self, game_state):
        # load board, snakes, and food
        self.board = game_state['board']
        self.my_snake_id = game_state['you']['id']
        self.width = self.board['width']
        self.height = self.board['height']
        self.food = {(f['x'], f['y']) for f in self.board['food']}
        self.snakes = {s['id']: s for s in self.board['snakes']}

        # detect which snakes just ate
        curr = {sid: len(s['body']) for sid, s in self.snakes.items()}
        self.just_ate = {sid: curr[sid] > self.prev_lengths.get(sid, 0) for sid in curr}
        self.prev_lengths = curr

        # build risk grid
        self.build_probability_grid()

    # 1) build risk grid of occupied and likely cells
    def build_probability_grid(self, debug=False):
        W, H = self.width, self.height
        grid = np.zeros((H, W))
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]

        # mark bodies as blocked
        for sid, s in self.snakes.items():
            parts = [(pt['x'],pt['y']) for pt in s['body'][:-1]]
            for x,y in parts:
                grid[y,x] = 1.0
            # block tail if just ate
            tx, ty = s['body'][-1]['x'], s['body'][-1]['y']
            if self.just_ate.get(sid):
                grid[ty,tx] = 1.0

        self.grid = grid

        # spread opponent move probabilities
        for sid, s in self.snakes.items():
            if sid == self.my_snake_id:
                continue
            hx, hy = s['body'][0]['x'], s['body'][0]['y']
            neck = tuple(s['body'][1].values())
            cands = []
            for dx,dy in dirs:
                nx, ny = hx+dx, hy+dy
                if not (0<=nx<W and 0<=ny<H):
                    continue
                if (nx,ny) == neck:
                    continue
                if grid[ny,nx] >= 1.0:
                    continue
                cands.append((nx,ny))
            if not cands:
                continue

            weights = []
            for nx,ny in cands:
                dist = min((abs(nx-fx)+abs(ny-fy)) for fx,fy in self.food) if self.food else W+H
                w_dist = 1.0/(dist+1)
                w_reach = self.expected_reach(nx,ny)
                weights.append(w_dist * w_reach)
            total = sum(weights)
            for (nx,ny),w in zip(cands,weights):
                p = (w/total) if total>0 else 1.0/len(cands)
                grid[ny,nx] = max(grid[ny,nx], p)

        self.grid = grid
        if debug:
            for y in range(H-1,-1,-1):
                print(' '.join(f'{v:.2f}' for v in grid[y]))

    # 2) A* with risk cost
    def expected_path_cost(self, start, goal, alpha=5.0):
        W, H = self.width, self.height
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

        open_set = [(h(start,goal), start)]
        g = {start: 0.0}
        while open_set:
            _, curr = heapq.heappop(open_set)
            if curr == goal:
                return g[curr]
            for dx,dy in dirs:
                nx, ny = curr[0]+dx, curr[1]+dy
                if not (0<=nx<W and 0<=ny<H):
                    continue
                p = self.grid[ny,nx]
                if p >= 1.0:
                    continue
                cost = 1.0 + alpha * p
                ng = g[curr] + cost
                if ng < g.get((nx,ny), float('inf')):
                    g[(nx,ny)] = ng
                    heapq.heappush(open_set, (ng + h((nx,ny),goal), (nx,ny)))
        return None

    # 3) flood-fill expected reach
    def expected_reach(self, sx, sy):
        W, H = self.width, self.height
        if not (0<=sx<W and 0<=sy<H):
            return 0.0
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        reach = np.zeros((H, W))
        reach[sy, sx] = 1.0
        q = deque([(sx,sy)])
        while q:
            x,y = q.popleft()
            base = reach[y,x]
            for dx,dy in dirs:
                nx, ny = x+dx, y+dy
                if not (0<=nx<W and 0<=ny<H):
                    continue
                p = self.grid[ny,nx]
                if p >= 1.0:
                    continue
                npv = base * (1.0 - p)
                if npv > reach[ny,nx]:
                    reach[ny,nx] = npv
                    q.append((nx,ny))
        return reach.sum()

    # 4) get safe moves
    def get_Safe_Moves(self):
        head = self.snakes[self.my_snake_id]['body'][0]
        x, y = head['x'], head['y']
        dirs = {'up':(0,1),'down':(0,-1),'left':(-1,0),'right':(1,0)}
        opts = []
        for m,(dx,dy) in dirs.items():
            nx, ny = x+dx, y+dy
            if 0<=nx<self.width and 0<=ny<self.height and self.grid[ny,nx] < 1.0:
                opts.append((m, self.grid[ny,nx]))
        if not opts:
            return []
        zeros = [m for m,p in opts if p == 0.0]
        if zeros:
            return zeros
        minp = min(p for _,p in opts)
        return [m for m,p in opts if p == minp]

    # 5a) food score
    def get_Food_Score(self, moves):
        if not self.food:
            return {m: 0.0 for m in moves}
        hx, hy = self.snakes[self.my_snake_id]['body'][0].values()
        best0 = min((self.expected_path_cost((hx,hy),f) or math.inf) for f in self.food)
        sc = {}
        for m in moves:
            dx,dy = {'up':(0,1),'down':(0,-1),'left':(-1,0),'right':(1,0)}[m]
            nx, ny = hx+dx, hy+dy
            if (nx,ny) in self.food:
                sc[m] = best0 + 10.0
            else:
                best1 = min((self.expected_path_cost((nx,ny),f) or math.inf) for f in self.food)
                sc[m] = best0 - best1
        return sc

    # 5b) space score
    def get_Space_Score(self, moves):
        hx, hy = self.snakes[self.my_snake_id]['body'][0].values()
        base = self.expected_reach(hx,hy)
        norm = 10.0 / (self.width * self.height)
        sc = {}
        for m in moves:
            dx,dy = {'up':(0,1),'down':(0,-1),'left':(-1,0),'right':(1,0)}[m]
            nx, ny = hx+dx, hy+dy
            sc[m] = (self.expected_reach(nx,ny) - base) * norm
        return sc

    # 5c) enemy space score
    def get_Enemy_Space_Score(self, moves):
        hx, hy = self.snakes[self.my_snake_id]['body'][0].values()
        others = [s['body'][0] for sid,s in self.snakes.items() if sid != self.my_snake_id]
        if not others:
            return {m: 0.0 for m in moves}
        ox, oy = min(others, key=lambda h: abs(h['x']-hx) + abs(h['y']-hy)).values()
        norm = 10.0 / (self.width * self.height)
        sc = {}
        for m in moves:
            dx,dy = {'up':(0,1),'down':(0,-1),'left':(-1,0),'right':(1,0)}[m]
            nx, ny = hx+dx, hy+dy
            saved = self.grid.copy()
            self.grid[ny,nx] = 1.0
            sc[m] = -self.expected_reach(ox,oy) * norm
            self.grid = saved
        return sc

    # 6) depth-limited lookahead
    def lookahead_move(self, depth=4):
        # try each safe root move
        opts = self.get_Safe_Moves()
        print(f"Root safe moves: {opts}")
        if not opts:
            return random.choice(['up','down','left','right'])
        best, choice = -math.inf, None
        for m in opts:
            v = self._search_value(self._clone_state(), m, depth, path=[m])
            print(f"Move {m} -> score = {v}")
            if v > best:
                best, choice = v, m
        print(f"Chosen move: {choice} with score = {best}")
        return choice

    # recursive search with path tracking
    def _search_value(self, state, move, depth, path=None):
        # simulate own move; record death as heavy penalty
        if path is None:
            path = [move]
        alive = self._apply_my_move(state, move)
        if not alive:
            print(f"Explored move: {' -> '.join(path)} score = -1000000")
            return -1e6
        # simulate opponents and rebuild grid
        self._apply_opponents(state)
        self._load_state_from_sim(state)
        self.build_probability_grid()
        # get safe moves
        safe = self.get_Safe_Moves()
        if not safe:
            print(f"Explored move: {' -> '.join(path)} score = -1000000")
            return -1e6
        # leaf: compute and report score
        if depth == 1:
            fs = self.get_Food_Score(safe)
            ss = self.get_Space_Score(safe)
            es = self.get_Enemy_Space_Score(safe)
            score = max((fs[m] + 4.0*ss[m] + 2.2*es[m]) for m in safe)
            print(f"Explored move: {' -> '.join(path)} score = {score}")
            return score
        # deepen search
        best_val = -math.inf
        for m in safe:
            val = self._search_value(self._clone_state(state), m, depth-1, path + [m])
            best_val = max(best_val, val)
        return best_val

    # clone state for simulation
    def _clone_state(self, state=None):
        base = state or {
            'snakes': self.snakes,
            'food': set(self.food),
            'prev_lengths': dict(self.prev_lengths),
            'width': self.width,
            'height': self.height
        }
        return copy.deepcopy(base)

    # apply own move
    def _apply_my_move(self, state, move):
        dirs = {'up':(0,1),'down':(0,-1),'left':(-1,0),'right':(1,0)}
        me = state['snakes'][self.my_snake_id]
        hx, hy = me['body'][0]['x'], me['body'][0]['y']
        dx, dy = dirs[move]
        nx, ny = hx+dx, hy+dy
        # wall or body collision
        if not (0<=nx<state['width'] and 0<=ny<state['height']):
            return False
        for s in state['snakes'].values():
            if any(pt['x']==nx and pt['y']==ny for pt in s['body']):
                return False
        me['body'].insert(0, {'x': nx, 'y': ny})
        if (nx, ny) not in state['food']:
            me['body'].pop()
        else:
            state['food'].remove((nx, ny))
        return True

    # apply opponents' moves
    def _apply_opponents(self, state):
        W, H = state['width'], state['height']
        for sid, s in state['snakes'].items():
            if sid == self.my_snake_id:
                continue
            hx, hy = s['body'][0]['x'], s['body'][0]['y']
            cands = []
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = hx+dx, hy+dy
                if 0<=nx<W and 0<=ny<H and self.grid[ny,nx]<1.0:
                    cands.append((nx, ny, self.grid[ny,nx]))
            if not cands:
                continue
            bx, by, _ = max(cands, key=lambda t: t[2])
            s['body'].insert(0, {'x': bx, 'y': by})
            if not self.just_ate.get(sid, False):
                s['body'].pop()

    # load simulated state into AI
    def _load_state_from_sim(self, state):
        self.snakes = state['snakes']
        self.food = set(state['food'])
        self.prev_lengths = state['prev_lengths']

    def get_Next_Move(self, debug=True):
        return self.lookahead_move(depth=4)
