# agents/shortened_hamiltonian.py
import numpy as np
from collections import deque
from agents.hamiltonian import HamiltonianAgent

class ShortenedHamiltonianAgent(HamiltonianAgent):
    """
    Агент, следующий по гамильтонову циклу, но делающий безопасные сокращения к еде.
    Наследует базовую логику цикла, добавляя эвристику безопасного среза.
    """
    def __init__(self, grid_size=10):
        super().__init__(grid_size)

    def get_action(self, env):
        head = env.snake[0]
        food = env.food

        # 1. Попытка найти безопасный срез к еде
        if food is not None:
            path_to_food = self._find_path_bfs(head, food, env.snake)
            if path_to_food and self._check_safety(path_to_food, env.snake):
                next_step = path_to_food[0]
                return self._get_direction(head, next_step)

        # 2. Если срез невозможен или опасен — возвращаемся к циклу
        head_idx = self.pos_to_idx.get(head, 0)
        cycle_next = self.path[(head_idx + 1) % len(self.path)]
        return self._get_direction(head, cycle_next)

    def _find_path_bfs(self, start, target, snake_body):
        """BFS поиск кратчайшего пути до еды (хвост не считается препятствием)."""
        obstacles = set(snake_body[:-1])
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path[1:]  # Возвращаем путь без стартовой клетки

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + dr, curr[1] + dc)
                if (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size
                        and nxt not in obstacles and nxt not in visited):
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def _check_safety(self, path_to_food, snake_body):
        """
        Виртуальная проверка: после поедания еды останется ли путь от новой головы к хвосту?
        Инвариант связности: если голова может достичь хвоста, агент не окажется в тупике.
        """
        virtual_head = path_to_food[-1]  # == food
        # Тело после еды: еда + старое тело без хвоста (хвост сдвинется на следующем шаге)
        virtual_body = [virtual_head] + snake_body[:-1]
        virtual_tail = snake_body[-1]

        obstacles = set(virtual_body)
        queue = deque([virtual_head])
        visited = {virtual_head}

        while queue:
            curr = queue.popleft()
            if curr == virtual_tail:
                return True  # Путь к хвосту существует → срез безопасен

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + dr, curr[1] + dc)
                if (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size
                        and nxt not in obstacles and nxt not in visited):
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def _get_direction(self, start, target):
        dr = target[0] - start[0]
        dc = target[1] - start[1]
        if dr == -1: return 0  # UP
        if dr == 1:  return 1  # DOWN
        if dc == -1: return 2  # LEFT
        if dc == 1:  return 3  # RIGHT
        return 0  # fallback