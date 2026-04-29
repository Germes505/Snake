# agents/hamiltonian.py
import numpy as np


class HamiltonianAgent:
    """
    Строгий Гамильтонов цикл для чётных полей (10x10, 20x20).
    Паттерн 'Гребёнка': Зигзаг по всему полю (кроме 1-го столбца),
    затем спуск вниз и подъём по 1-му столбцу для замыкания цикла.
    """

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.path = self._build_cycle()

        # Словарь для быстрого поиска: (row, col) -> индекс в пути
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.path)}

        # Проверка валидности при инициализации
        if len(self.path) != grid_size * grid_size:
            raise ValueError(f"Ошибка цикла: длина {len(self.path)} != {grid_size ** 2}")
        if not self._validate_adjacency():
            raise ValueError("Ошибка цикла: найдены разрывы в пути!")

    def _build_cycle(self):
        """
        Генерирует замкнутый путь.
        """
        path = []

        # 1. Зигзаг по всем рядам, НО пропускаем столбец 0 (колонка 0 — это "позвоночник")
        for r in range(self.grid_size):
            if r % 2 == 0:
                # Чётный ряд: Слева направо (от 1 до конца)
                for c in range(1, self.grid_size):
                    path.append((r, c))
            else:
                # Нечётный ряд: Справа налево (от конца до 1)
                for c in range(self.grid_size - 1, 0, -1):
                    path.append((r, c))

        # На данный момент мы находимся в (grid_size-1, 1) — правее левого нижнего угла

        # 2. Шаг влево в "позвоночник" (колонка 0)
        path.append((self.grid_size - 1, 0))

        # 3. Подъём вверх по колонке 0 до самого верха (0,0)
        for r in range(self.grid_size - 2, -1, -1):
            path.append((r, 0))

        return path

    def _validate_adjacency(self):
        """Проверяет, что расстояние между соседними точками пути равно 1."""
        for i in range(len(self.path)):
            curr = self.path[i]
            next_pos = self.path[(i + 1) % len(self.path)]
            dist = abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1])
            if dist != 1:
                return False
        return True

    def get_action(self, env):
        """Возвращает направление (0-3) к следующей клетке цикла."""
        head = env.snake[0]
        idx = self.pos_to_idx.get(head)

        # Если голова не найдена (теоретически невозможно при правильной работе),
        # возвращаем текущее направление, чтобы не крашнуться
        if idx is None:
            return env.direction

        next_idx = (idx + 1) % len(self.path)
        next_pos = self.path[next_idx]

        dr = next_pos[0] - head[0]
        dc = next_pos[1] - head[1]

        if dr == -1: return 0  # UP
        if dr == 1:  return 1  # DOWN
        if dc == -1: return 2  # LEFT
        if dc == 1:  return 3  # RIGHT
        return env.direction

    def get_start_direction(self, pos):
        """
        Вспомогательный метод: какое направление должна иметь змейка
        в позиции pos, чтобы двигаться по циклу?
        Нужно для корректного старта в test_hamiltonian.py
        """
        idx = self.pos_to_idx.get(pos)
        if idx is None: return 0  # Fallback

        next_idx = (idx + 1) % len(self.path)
        next_pos = self.path[next_idx]

        dr = next_pos[0] - pos[0]
        dc = next_pos[1] - pos[1]

        if dr == -1: return 0  # UP
        if dr == 1:  return 1  # DOWN
        if dc == -1: return 2  # LEFT
        if dc == 1:  return 3  # RIGHT
        return 0