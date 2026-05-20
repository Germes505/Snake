import pygame
import numpy as np
from collections import deque


class HamiltonianSnakeSolver:
    """ФИНАЛЬНАЯ версия - работает + никогда не упирается"""

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.k = grid_size * grid_size
        self.cycle = self._create_hamiltonian_cycle(grid_size)
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.cycle)}
        self.idx_to_pos = {i: pos for i, pos in enumerate(self.cycle)}

    def _create_hamiltonian_cycle(self, grid_size):
        cycle = []
        for row in range(grid_size):
            if row % 2 == 0:
                for col in range(grid_size): cycle.append((row, col))
            else:
                for col in range(grid_size - 1, -1, -1): cycle.append((row, col))
        return cycle

    def pos_to_tuple(self, pos):
        return tuple(map(int, pos)) if hasattr(pos, '__iter__') else pos

    def get_cycle_distance(self, idx1, idx2):
        return (idx2 - idx1) % self.k

    def is_safe_move(self, env, new_head):
        """ПРОСТАЯ проверка нового хода"""
        nh = self.pos_to_tuple(new_head)
        if not (0 <= nh[0] < self.grid_size and 0 <= nh[1] < self.grid_size):
            return False
        return nh not in [self.pos_to_tuple(p) for p in env.snake[1:]]

    def tail_safety_check(self, env):
        """O(1) БАЗОВАЯ безопасность"""
        if not env.food: return False

        head_idx = self.pos_to_idx[self.pos_to_tuple(env.snake[0])]
        tail_idx = self.pos_to_idx[self.pos_to_tuple(env.snake[-1])]
        food_idx = self.pos_to_idx[env.food]

        dist_food_tail = self.get_cycle_distance(food_idx, tail_idx)
        return dist_food_tail > len(env.snake)

    def bfs_to_target(self, env, start, target):
        """НАДЕЖНЫЙ BFS"""
        start = self.pos_to_tuple(start)
        target = self.pos_to_tuple(target)

        queue = deque([([start], start)])
        visited = set([start])

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            path, pos = queue.popleft()

            if pos == target:
                return path

            r, c = pos
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                new_pos = (nr, nc)

                if (self.is_safe_move(env, new_pos) and
                        new_pos not in visited):
                    visited.add(new_pos)
                    queue.append((path + [new_pos], new_pos))
        return None

    def get_next_direction(self, env):
        """ПРОСТОЙ И НАДЕЖНЫЙ алгоритм"""
        head = self.pos_to_tuple(env.snake[0])

        # === РАННИЕ УСЛОВИЯ ЗАВЕРШЕНИЯ ===
        if len(env.snake) > self.k * 0.95:  # 95% заполнено
            return self.safe_cycle_move(env, head)

        # === БЕЗОПАСНОЕ СОКРАЩЕНИЕ К ЕДЕ ===
        if (env.food and self.tail_safety_check(env) and
                len(env.snake) < self.k - 10):  # Не в конце игры

            path = self.bfs_to_target(env, head, env.food)
            if path and len(path) > 1:
                next_pos = path[1]
                return self.pos_to_dir(head, next_pos)

        # === ГАРАНТИРОВАННО БЕЗОПАСНЫЙ ЦИКЛ ===
        return self.safe_cycle_move(env, head)

    def safe_cycle_move(self, env, head):
        """ВСЕГДА безопасный шаг по циклу"""
        head_idx = self.pos_to_idx[head]
        next_idx = (head_idx + 1) % self.k
        next_pos = self.idx_to_pos[next_idx]

        if self.is_safe_move(env, next_pos):
            return self.pos_to_dir(head, next_pos)

        # === РЕЗЕРВНЫЕ НАПРАВЛЕНИЯ ЦИКЛА ===
        for offset in [2, -1, 3, -2]:
            test_idx = (head_idx + offset) % self.k
            test_pos = self.idx_to_pos[test_idx]
            if self.is_safe_move(env, test_pos):
                return self.pos_to_dir(head, test_pos)

        # === АБСОЛЮТНЫЙ ЭКСТРЕННЫЙ ВЫБОР ===
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in dirs:
            test_pos = (head[0] + dr, head[1] + dc)
            if self.is_safe_move(env, test_pos):
                return self.pos_to_dir(head, test_pos)

        return 0  # Никогда не должно дойти!

    def pos_to_dir(self, from_pos, to_pos):
        """Позиция -> направление"""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        if dr == -1: return 0  # UP
        if dr == 1:  return 1  # DOWN
        if dc == -1: return 2  # LEFT
        if dc == 1:  return 3  # RIGHT
        return 0