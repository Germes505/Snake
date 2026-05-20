# agents/greedy.py
"""
Greedy Agent для SnakeEnv — точная адаптация оригинального GreedySolver.
Исправлены: проходимость хвоста, симуляция роста, индексация расширения пути.
"""

from collections import deque
from typing import List, Tuple, Set, Optional
import random


class GreedyAgent:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    # ═══════════════════════════════════════════════════════
    # 🎯 ГЛАВНЫЙ МЕТОД
    # ═══════════════════════════════════════════════════════
    def get_action(self, env) -> int:
        head = env.snake[0]
        # ✅ Хвост проходим! Исключаем его из препятствий
        obstacles = set(env.snake[1:-1]) if len(env.snake) > 2 else set()

        # 🔹 ШАГ 1: Кратчайший путь к еде
        path_to_food = self._bfs(head, env.food, obstacles)

        if path_to_food:
            # 🔹 ШАГ 2: Симуляция поедания
            simulated_snake = self._simulate(env.snake, path_to_food, env.food)

            # Если поле заполнено → идём смело
            if len(simulated_snake) >= self.grid_size ** 2:
                return self._pos_to_direc(head, path_to_food[0])

            # 🔹 ШАГ 3: Проверка безопасности (путь до хвоста после еды)
            tail_after = simulated_snake[-1]
            obs_after = set(simulated_snake[1:-1]) if len(simulated_snake) > 2 else set()
            path_to_tail = self._bfs(simulated_snake[0], tail_after, obs_after)

            if path_to_tail:
                path_to_tail = self._extend_path(path_to_tail, obs_after)
                if len(path_to_tail) > 1:
                    return self._pos_to_direc(head, path_to_food[0])

        # 🔹 ШАГ 4: Режим выживания (следовать за текущим хвостом)
        current_tail = env.snake[-1]
        path_to_tail = self._bfs(head, current_tail, obstacles)
        if path_to_tail:
            path_to_tail = self._extend_path(path_to_tail, obstacles)
            if len(path_to_tail) > 1:
                return self._pos_to_direc(head, path_to_tail[0])

        # 🔹 ШАГ 5: Аварийный ход
        return self._emergency(env, head, obstacles)

    # ═══════════════════════════════════════════════════════
    # 🔍 BFS (кратчайший путь)
    # ═══════════════════════════════════════════════════════
    def _bfs(self, start: Tuple[int, int], goal: Optional[Tuple[int, int]],
             obstacles: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        if goal is None or start == goal:
            return []

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            cur, path = queue.popleft()

            # Соседи: сначала пробуем продолжить в текущем направлении (если есть)
            adjs = []
            for direc, (dr, dc) in enumerate(self.DIRECTIONS):
                nxt = (cur[0] + dr, cur[1] + dc)
                if 0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size:
                    adjs.append((direc, nxt))

            random.shuffle(adjs)
            # Приоритет "прямо" можно добавить, но для 8x8 это не критично.

            for direc, nxt in adjs:
                if nxt == goal or nxt not in obstacles:
                    if nxt not in visited:
                        new_path = path + [nxt]
                        if nxt == goal:
                            return new_path[1:]  # Без стартовой клетки
                        visited.add(nxt)
                        queue.append((nxt, new_path))
        return None

    # ═══════════════════════════════════════════════════════
    # 📏 Расширение пути зигзагами (эвристика длинного пути)
    # ═══════════════════════════════════════════════════════
    def _extend_path(self, path: List[Tuple[int, int]],
                     obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not path or len(path) < 2:
            return path

        # Все занятые клетки: препятствия + исходный путь
        occupied = set(obstacles)
        for p in path:
            occupied.add(p)

        extended = list(path)
        i = 0
        while i < len(extended) - 1:
            cur = extended[i]
            nxt = extended[i + 1]
            dr, dc = nxt[0] - cur[0], nxt[1] - cur[1]

            # Перпендикулярные направления для "петли"
            tests = [(-1, 0), (1, 0)] if dc != 0 else [(0, -1), (0, 1)]

            inserted = False
            for off_r, off_c in tests:
                c1 = (cur[0] + off_r, cur[1] + off_c)
                c2 = (nxt[0] + off_r, nxt[1] + off_c)

                # Обе клетки свободны?
                if (0 <= c1[0] < self.grid_size and 0 <= c1[1] < self.grid_size and
                        0 <= c2[0] < self.grid_size and 0 <= c2[1] < self.grid_size and
                        c1 not in occupied and c2 not in occupied):
                    # Вставляем петлю
                    occupied.update([c1, c2])
                    extended.insert(i + 1, c1)
                    extended.insert(i + 2, c2)
                    inserted = True
                    break

            # ✅ Корректное продвижение индекса после вставки
            i += 3 if inserted else 1

        return extended

    # ═══════════════════════════════════════════════════════
    # 🎮 Симуляция движения
    # ═══════════════════════════════════════════════════════
    def _simulate(self, snake: List[Tuple[int, int]],
                  path: List[Tuple[int, int]],
                  food: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
        sim = list(snake)
        for pos in path:
            sim.insert(0, pos)
            if pos == food:
                food = None  # Еда съедена, больше не растём
            else:
                sim.pop()  # Хвост сдвигается
        return sim

    # ═══════════════════════════════════════════════════════
    # 🆘 Аварийный ход
    # ═══════════════════════════════════════════════════════
    def _emergency(self, env, head: Tuple[int, int],
                   obstacles: Set[Tuple[int, int]]) -> int:
        best_direc = env.direction
        best_score = -1

        for direc, (dr, dc) in enumerate(self.DIRECTIONS):
            nxt = (head[0] + dr, head[1] + dc)
            if not (0 <= nxt[0] < env.grid_size and 0 <= nxt[1] < env.grid_size):
                continue
            if nxt in obstacles:
                continue

            # Оценка: количество свободных соседей (безопасность) + дистанция
            free_neighbors = 0
            for nr, nc in self.DIRECTIONS:
                nn = (nxt[0] + nr, nxt[1] + nc)
                if (0 <= nn[0] < env.grid_size and 0 <= nn[1] < env.grid_size and
                        nn not in obstacles):
                    free_neighbors += 1

            dist = abs(nxt[0] - env.food[0]) + abs(nxt[1] - env.food[1]) if env.food else 0
            score = free_neighbors * 100 - dist  # Приоритет открытому пространству

            if score > best_score or (score == best_score and direc == env.direction):
                best_score = score
                best_direc = direc
        return best_direc

    def _pos_to_direc(self, cur: Tuple[int, int], nxt: Tuple[int, int]) -> int:
        dr, dc = nxt[0] - cur[0], nxt[1] - cur[1]
        if dr == -1: return self.UP
        if dr == 1:  return self.DOWN
        if dc == -1: return self.LEFT
        if dc == 1:  return self.RIGHT
        return self.UP