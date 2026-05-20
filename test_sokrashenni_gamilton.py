#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import copy
import pygame
from enum import IntEnum, unique
from collections import deque

@unique
class Direc(IntEnum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    @staticmethod
    def opposite(d):
        return {Direc.UP: Direc.DOWN, Direc.DOWN: Direc.UP,
                Direc.LEFT: Direc.RIGHT, Direc.RIGHT: Direc.LEFT}[d]

@unique
class PointType(IntEnum):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    HEAD_L = 3
    HEAD_U = 4
    HEAD_R = 5
    HEAD_D = 6
    BODY_HOR = 7
    BODY_VER = 8

class Pos:
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    def __str__(self):
        return f'Pos({self._x},{self._y})'

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, Pos) and self._x == other._x and self._y == other._y

    def __hash__(self):
        return hash((self._x, self._y))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @staticmethod
    def manhattan_dist(p1, p2):
        return abs(p1._x - p2._x) + abs(p1._y - p2._y)

    def direc_to(self, adj_pos):
        if self._x == adj_pos._x:
            return Direc.LEFT if self._y - adj_pos._y == 1 else Direc.RIGHT
        elif self._y == adj_pos._y:
            return Direc.UP if self._x - adj_pos._x == 1 else Direc.DOWN
        return Direc.NONE

    def adj(self, direc):
        if direc == Direc.UP:    return Pos(self._x - 1, self._y)
        if direc == Direc.DOWN:  return Pos(self._x + 1, self._y)
        if direc == Direc.LEFT:  return Pos(self._x, self._y - 1)
        if direc == Direc.RIGHT: return Pos(self._x, self._y + 1)
        return None

    def all_adj(self):
        return [self.adj(d) for d in Direc if d != Direc.NONE]


class _MapPoint:
    def __init__(self):
        self.type = PointType.EMPTY


class Map:
    def __init__(self, rows, cols):
        self.num_rows = rows
        self.num_cols = cols
        self.grid = [[_MapPoint() for _ in range(cols)] for _ in range(rows)]
        self.food = None
        self._init_walls()

    def _init_walls(self):
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if r == 0 or r == self.num_rows - 1 or c == 0 or c == self.num_cols - 1:
                    self.grid[r][c].type = PointType.WALL

    def point(self, pos):
        return self.grid[pos.x][pos.y]

    def is_safe(self, pos):
        if 0 <= pos.x < self.num_rows and 0 <= pos.y < self.num_cols:
            t = self.point(pos).type
            return t == PointType.EMPTY or t == PointType.FOOD
        return False

    def is_full(self):
        playable_cells = (self.num_rows - 2) * (self.num_cols - 2)
        return len(snake_global.body) >= playable_cells

    def create_rand_food(self, exclude=()):
        available = []
        for r in range(1, self.num_rows - 1):
            for c in range(1, self.num_cols - 1):
                p = Pos(r, c)
                if self.point(p).type == PointType.EMPTY and p not in exclude:
                    available.append(p)
        if not available:
            return None
        self.food = random.choice(available)
        self.point(self.food).type = PointType.FOOD
        return self.food


class Snake:
    def __init__(self, game_map, init_direc, init_bodies):
        self.map = game_map
        self.direc = init_direc
        self.direc_next = init_direc
        self.body = list(init_bodies)  # Head at index 0
        self.steps = 0
        self.dead = False
        self._sync_map()

    def _sync_map(self):
        for i, p in enumerate(self.body):
            if i == 0:
                self.map.point(p).type = PointType.HEAD_R
            else:
                self.map.point(p).type = PointType.BODY_HOR

    def head(self):
        return self.body[0]

    def tail(self):
        return self.body[-1]

    def len(self):
        return len(self.body)

    def copy(self):
        m_copy = copy.deepcopy(self.map)
        s_copy = Snake(m_copy, self.direc, list(self.body))
        s_copy.direc_next = self.direc_next
        s_copy.steps = self.steps
        s_copy.dead = self.dead
        return s_copy, m_copy

    def move_path(self, path):
        for d in path:
            self._move_single(d)

    def _move_single(self, d):
        if self.dead: return
        self.direc = d
        new_head = self.head().adj(d)

        if not self.map.is_safe(new_head) and new_head != self.map.food:
            self.dead = True
            return

        self.body.insert(0, new_head)
        old_tail = self.tail()

        if new_head == self.map.food:
            self.map.point(new_head).type = PointType.BODY_HOR
            self.map.create_rand_food(exclude=self.body)
        else:
            self.body.pop()
            self.map.point(old_tail).type = PointType.EMPTY
            self.map.point(new_head).type = PointType.HEAD_R
            # Fix body type for new head
            if len(self.body) > 1:
                self.map.point(self.body[1]).type = PointType.BODY_HOR

        self.steps += 1

    def reset(self):
        # Clear old snake from map
        for p in self.body:
            self.map.point(p).type = PointType.EMPTY
        self.body = list(init_bodies)
        self.direc = init_direc
        self.direc_next = init_direc
        self.steps = 0
        self.dead = False
        self._sync_map()
        self.map.create_rand_food(exclude=self.body)

class BaseSolver:
    def __init__(self, snake):
        self._snake = snake
        self._map = snake.map

    @property
    def map(self): return self._map

    @property
    def snake(self): return self._snake

    @snake.setter
    def snake(self, val):
        self._snake = val
        self._map = val.map

    def next_direc(self):
        return NotImplemented

    def close(self): pass


class _TableCell:
    def __init__(self): self.reset()

    def reset(self):
        self.parent = None
        self.dist = sys.maxsize
        self.visit = False


class PathSolver(BaseSolver):
    def __init__(self, snake):
        super().__init__(snake)
        self._table = [[_TableCell() for _ in range(snake.map.num_cols)]
                       for _ in range(snake.map.num_rows)]

    @property
    def table(self):
        return self._table

    def shortest_path_to_food(self):
        return self.path_to(self.map.food, "shortest")

    def longest_path_to_tail(self):
        return self.path_to(self.snake.tail(), "longest")

    def path_to(self, des, path_type):
        ori_type = self.map.point(des).type
        self.map.point(des).type = PointType.EMPTY
        path = self.shortest_path_to(des) if path_type == "shortest" else self.longest_path_to(des)
        self.map.point(des).type = ori_type
        return path

    def shortest_path_to(self, des):
        self._reset_table()
        head = self.snake.head()
        self._table[head.x][head.y].dist = 0
        queue = deque([head])

        while queue:
            cur = queue.popleft()
            if cur == des:
                return self._build_path(head, des)

            if cur == head:
                first_direc = self.snake.direc
            else:
                first_direc = self._table[cur.x][cur.y].parent.direc_to(cur)

            adjs = cur.all_adj()
            random.shuffle(adjs)
            for i, pos in enumerate(adjs):
                if first_direc == cur.direc_to(pos):
                    adjs[0], adjs[i] = adjs[i], adjs[0]
                    break

            for pos in adjs:
                if self._is_valid(pos):
                    adj_cell = self._table[pos.x][pos.y]
                    if adj_cell.dist == sys.maxsize:
                        adj_cell.parent = cur
                        adj_cell.dist = self._table[cur.x][cur.y].dist + 1
                        queue.append(pos)
        return deque()

    def longest_path_to(self, des):
        path = self.shortest_path_to(des)
        if not path: return deque()

        self._reset_table()
        cur = head = self.snake.head()
        self._table[cur.x][cur.y].visit = True
        for direc in path:
            cur = cur.adj(direc)
            self._table[cur.x][cur.y].visit = True

        idx, cur = 0, head
        while True:
            cur_direc = path[idx]
            nxt = cur.adj(cur_direc)
            tests = [Direc.UP, Direc.DOWN] if cur_direc in (Direc.LEFT, Direc.RIGHT) else [Direc.LEFT, Direc.RIGHT]

            extended = False
            for test_direc in tests:
                cur_test = cur.adj(test_direc)
                nxt_test = nxt.adj(test_direc)
                if self._is_valid(cur_test) and self._is_valid(nxt_test):
                    self._table[cur_test.x][cur_test.y].visit = True
                    self._table[nxt_test.x][nxt_test.y].visit = True
                    path.insert(idx, test_direc)
                    path.insert(idx + 2, Direc.opposite(test_direc))
                    extended = True
                    break

            if not extended:
                cur = nxt
                idx += 1
                if idx >= len(path): break
        return path

    def _reset_table(self):
        for row in self._table:
            for col in row: col.reset()

    def _build_path(self, src, des):
        path = deque()
        tmp = des
        while tmp != src:
            parent = self._table[tmp.x][tmp.y].parent
            path.appendleft(parent.direc_to(tmp))
            tmp = parent
        return path

    def _is_valid(self, pos):
        return self.map.is_safe(pos) and not self._table[pos.x][pos.y].visit


class GreedySolver(BaseSolver):
    def __init__(self, snake):
        super().__init__(snake)
        self._path_solver = PathSolver(snake)

    def next_direc(self):
        playable_cells = (self.map.num_rows - 2) * (self.map.num_cols - 2)
        fill_ratio = len(self.snake.body) / playable_cells
        print(fill_ratio)
        if fill_ratio > 0.98:
            self._path_solver.snake = self.snake
            path_to_food = self._path_solver.shortest_path_to_food()
            if path_to_food:
                return path_to_food[0]  # Идём к еде даже если умрём!

        # Create a virtual snake
        s_copy, m_copy = self.snake.copy()

        # Step 1
        self._path_solver.snake = self.snake
        path_to_food = self._path_solver.shortest_path_to_food()

        if path_to_food:
            # Step 2
            s_copy.move_path(path_to_food)
            if m_copy.is_full():
                return path_to_food[0]

            # Step 3
            self._path_solver.snake = s_copy
            path_to_tail = self._path_solver.longest_path_to_tail()
            if len(path_to_tail) > 1:
                return path_to_food[0]

        # Step 4
        self._path_solver.snake = self.snake
        path_to_tail = self._path_solver.longest_path_to_tail()
        if len(path_to_tail) > 1:
            return path_to_tail[0]

        # Step 5
        head = self.snake.head()
        direc, max_dist = self.snake.direc, -1
        for adj in head.all_adj():
            if self.map.is_safe(adj):
                dist = Pos.manhattan_dist(adj, self.map.food)
                if dist > max_dist:
                    max_dist = dist
                    direc = head.direc_to(adj)
        return direc

init_direc = Direc.RIGHT
map_rows, map_cols = 20, 20
cell_size = 30
window_width, window_height = (map_cols + 2) * cell_size, (map_rows + 2) * cell_size
init_bodies = [Pos(4, 5), Pos(4, 4), Pos(4, 3)]
snake_global = None
solver_global = None


def init_game():
    global snake_global, solver_global
    game_map = Map(map_rows + 2, map_cols + 2)
    snake_global = Snake(game_map, init_direc, init_bodies)
    game_map.create_rand_food(exclude=snake_global.body)
    solver_global = GreedySolver(snake_global)


def draw(screen):
    screen.fill((20, 20, 20))

    # Стены
    for r in range(map_rows + 2):
        for c in range(map_cols + 2):
            if snake_global.map.point(Pos(r, c)).type == PointType.WALL:
                pygame.draw.rect(screen, (80, 80, 80), (c * cell_size, r * cell_size, cell_size, cell_size))

    # Еда
    if snake_global.map.food:
        f = snake_global.map.food
        pygame.draw.rect(screen, (255, 0, 0), (f.y * cell_size, f.x * cell_size, cell_size - 1, cell_size - 1))

    # Змейка
    for i, p in enumerate(snake_global.body):
        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        pygame.draw.rect(screen, color, (p.y * cell_size, p.x * cell_size, cell_size - 2, cell_size - 2))

    # HUD
    font = pygame.font.SysFont('consolas', 16)
    txt = font.render(f"Score: {snake_global.len() - 3} | Steps: {snake_global.steps}", True, (200, 200, 200))
    screen.blit(txt, (10, (map_rows + 1) * cell_size + 5))


def main():
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Snake - Greedy Solver")
    clock = pygame.time.Clock()
    init_game()

    running = True
    pause = False
    tick_count = 0
    TICK_RATE = 2000  # Скорость игры (кадров в секунду)

    while running:
        tick_count += clock.tick(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    init_game()
                    print("🔄 Рестарт")
                elif event.key == pygame.K_SPACE:
                    pause = not pause

        if not pause and not snake_global.dead and tick_count >= 1000 // TICK_RATE:
            tick_count = 0
            # Получить направление от солвера
            next_d = solver_global.next_direc()
            if next_d != Direc.NONE:
                snake_global._move_single(next_d)

            if snake_global.dead:
                print(f"💀 Смерть | Score: {snake_global.len()} | Steps: {snake_global.steps}")
            elif snake_global.map.is_full():
                print(f"🏆 ПОБЕДА | Поле заполнено! Steps: {snake_global.steps}")

        draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()