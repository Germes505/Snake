# agents/shortened_hamiltonian.py
from collections import deque
from agents.hamiltonian import HamiltonianAgent


class ShortenedHamiltonianAgent(HamiltonianAgent):
    """
    Финальная версия агента с геометрической проверкой безопасности.
     Решает проблему 'нет места для отхода' через проверку пространства хвоста.
    """

    def __init__(self, grid_size=10, debug=False):
        super().__init__(grid_size)
        self.debug = debug

    def get_action(self, env):
        head = env.snake[0]
        food = env.food

        if food is None:
            return self._get_cycle_dir(env)

        # 1. Волновой поиск пути к еде
        path = self._bfs_dynamic(head, food, env.snake)
        if not path:
            return self._get_cycle_dir(env)

        # 2. Геометрическая проверка безопасности
        if not self._is_geometrically_safe(path, env.snake):
            if self.debug: print("[] Срез геометрически опасен → цикл")
            return self._get_cycle_dir(env)

        if self.debug: print(f"[✅] Безопасный срез → {path[0]}")
        return self._get_dir(head, path[0])

    def _is_geometrically_safe(self, path, snake):
        """
        Тройная проверка, гарантирующая пространство для манёвра:
        1. Консервативный лимит длины (>60% поля → только цикл)
        2. Достижимость хвоста (head -> tail)
        3. Пространство хвоста (tail -> free_area > 15% поля)
        """
        # 1. Длинные змейки не рискуют. На поздних этапах цикл гарантирует 100% выживание.
        if len(snake) > self.grid_size * self.grid_size * 0.6:
            return False

        new_head = path[-1]
        # Виртуальное тело после еды: [еда] + старое тело без хвоста
        virtual_body = [new_head] + snake[:-1]
        virtual_tail = snake[-1]
        obstacles = set(virtual_body[:-1])  # Хвост не препятствие

        # 2. Голова должна видеть хвост (инвариант связности)
        if not self._can_reach(new_head, virtual_tail, obstacles):
            return False

        # 3. 🔑 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Хвост должен иметь пространство для отхода
        # Считаем, сколько клеток доступно из хвоста. Если < 15% поля → хвост заперт.
        tail_free_space = self._flood_fill_count(virtual_tail, obstacles)
        min_safe_space = (self.grid_size ** 2) * 0.15

        if tail_free_space < min_safe_space:
            return False  # Хвост в "кармане", отходить некуда

        return True

    def _bfs_dynamic(self, start, target, snake):
        """BFS с учётом динамики хвоста"""
        L = len(snake)
        body_map = {pos: i for i, pos in enumerate(snake)}
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            arrive_time = len(path)

            if curr == target:
                return path[1:]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + dr, curr[1] + dc)
                if not (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size):
                    continue
                if nxt in visited:
                    continue

                if nxt in body_map:
                    seg_idx = body_map[nxt]
                    body_num_from_tail = L - seg_idx
                    # Волна должна догнать освобождающееся тело
                    if arrive_time - 1 < body_num_from_tail:
                        continue

                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    def _can_reach(self, start, target, obstacles):
        queue, visited = deque([start]), {start}
        while queue:
            curr = queue.popleft()
            if curr == target: return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + dr, curr[1] + dc)
                if 0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size:
                    if nxt not in obstacles and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        return False

    def _flood_fill_count(self, start, obstacles):
        queue, visited = deque([start]), {start}
        count = 0
        while queue:
            curr = queue.popleft()
            count += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + dr, curr[1] + dc)
                if 0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size:
                    if nxt not in obstacles and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        return count

    def _get_cycle_dir(self, env):
        """Возврат на цикл с обходом тела"""
        head = env.snake[0]
        idx = self.pos_to_idx.get(head)
        if idx is None:
            idx = min(range(len(self.path)), key=lambda i:
            abs(self.path[i][0] - head[0]) + abs(self.path[i][1] - head[1]))

        target = self.path[(idx + 1) % len(self.path)]
        if target not in env.snake[:-1]:
            return self._get_dir(head, target)

        opposite = env.direction ^ 1
        dirs = [env.direction] + [d for d in [0, 1, 2, 3] if d != env.direction and d != opposite]
        for d in dirs:
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][d]
            nxt = (head[0] + dr, head[1] + dc)
            if 0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size:
                if nxt not in env.snake[:-1]:
                    return d
        return self._get_dir(head, target)

    def _get_dir(self, start, target):
        dr, dc = target[0] - start[0], target[1] - start[1]
        if dr == -1: return 0
        if dr == 1:  return 1
        if dc == -1: return 2
        if dc == 1:  return 3
        return 0