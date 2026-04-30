import collections
from typing import List, Tuple, Optional
from agents.hamiltonian import HamiltonianAgent


def is_valid_position(env, pos):
    """Ручная проверка валидности позиции"""
    x, y = pos
    if not (0 <= x < env.grid_size and 0 <= y < env.grid_size):
        return False
    if tuple(pos) in [tuple(p) for p in env.snake[1:]]:
        return False
    return True


class ShortenedHamiltonianAgent(HamiltonianAgent):
    def __init__(self, grid_size=None):
        super().__init__(grid_size) if grid_size else super().__init__()
        self.cycle_pos = 0
        self.next_action = None
        self.hamiltonian_cycle = self.path
        self.pos_to_idx = self.pos_to_idx

    def get_action(self, env) -> int:
        if len(env.snake) % 10 == 0:
            print(f"DEBUG | Длина:{len(env.snake)} | "
                  f"Shortcut:{self._safe_shortcut_to_food(env)}")

        if self._safe_shortcut_to_food(env):
            return self.next_action or 0

        return self._follow_cycle_with_bypass(env)  # ✅ Метод ДОЛЖЕН существовать!

    def _safe_shortcut_to_food(self, env) -> bool:
        """🔥 ПРОСТОЙ И БЕЗОПАСНЫЙ срез"""
        if len(env.snake) > 70:  # Фиксированный лимит для 10x10
            return False

        head = tuple(env.snake[0])
        food = tuple(env.food)

        # ТОЛЬКО прямой путь к еде (без сложной динамики)
        path = self._bfs_simple(env, head, food)
        if path and len(path) <= 15:  # Короткие срезы только!
            print(f"✅ БЕЗОПАСНЫЙ СРЕЗ: {len(path)} шагов")
            self.next_action = self._path_to_direction(path)
            return True
        return False

    def _bfs_simple(self, env, start, goal):
        """🔧 ПРОСТОЙ BFS без динамики тела"""
        from collections import deque
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([(start, [start])])
        visited = set([start])

        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path

            x, y = pos
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size and
                        new_pos not in visited and new_pos not in [tuple(p) for p in env.snake[1:]]):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        return None

    def _bfs_dynamic(self, env, start, goal):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = collections.deque([(start, [start], 1)])
        visited = set([start])

        body_numbers = {pos: i + 1 for i, pos in enumerate(reversed(env.snake))}

        while queue:
            pos, path, wave_time = queue.popleft()
            if pos == goal:
                return path

            x, y = pos
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)

                if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size and
                        new_pos not in visited):

                    body_num = body_numbers.get(new_pos, 0)
                    arrive_time = wave_time + 1
                    if body_num > 0 and arrive_time <= body_num:
                        continue

                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos], wave_time + 1))
        return None

    def _can_reach_new_tail(self, env, new_head):
        new_snake = [new_head] + env.snake[:-1]
        new_tail = tuple(new_snake[-1])

        visited = set()
        queue = collections.deque([new_head])
        visited.add(new_head)

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == new_tail:
                return True

            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                pos = (nx, ny)
                if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size and
                        pos not in visited and pos not in [tuple(p) for p in new_snake[1:]]):
                    visited.add(pos)
                    queue.append((nx, ny))
        return False

    def _is_geometrically_safe(self, env):
        tail = tuple(env.snake[-1])
        free = self._flood_fill_count(env, tail)
        return free >= 0.15 * env.grid_size * env.grid_size

    def _flood_fill_count(self, env, start):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = collections.deque([start])
        visited = set([start])
        count = 1

        while queue:
            x, y = queue.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                pos = (nx, ny)
                if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size and
                        pos not in visited and pos not in [tuple(p) for p in env.snake]):
                    visited.add(pos)
                    queue.append(pos)
                    count += 1
        return count

    def _follow_cycle_with_bypass(self, env):  # ✅ ОПРЕДЕЛЕН ЗДЕСЬ!
        """🔥 СЛЕДОВАНИЕ ПО ГАМИЛЬТОНОВУ ЦИКЛУ"""
        head = tuple(env.snake[0])
        head_idx = self.pos_to_idx.get(head)

        if head_idx is None:
            return self._find_safe_bypass(env, head)

        cycle_next = self.hamiltonian_cycle[(head_idx + 1) % len(self.hamiltonian_cycle)]

        if is_valid_position(env, cycle_next):
            self.cycle_pos = (self.cycle_pos + 1) % len(self.hamiltonian_cycle)
            return self._pos_to_direction(head, cycle_next)
        else:
            return self._find_safe_bypass(env, head)

    def _find_safe_bypass(self, env, head):
        """Временный безопасный обход"""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dirs:
            nx, ny = head[0] + dx, head[1] + dy
            if is_valid_position(env, (nx, ny)):
                return self._dxdy_to_action(dx, dy)
        return 0  # Экстренный UP

    def _path_to_direction(self, path):
        head, next_pos = path[0], path[1]
        return self._pos_to_direction(head, next_pos)

    def _pos_to_direction(self, from_pos, to_pos):
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        return self._dxdy_to_action(dx, dy)

    def _dxdy_to_action(self, dx, dy):
        if dx == -1: return 0  # UP
        if dx == 1: return 1  # DOWN
        if dy == -1: return 2  # LEFT
        if dy == 1: return 3  # RIGHT
        return 0