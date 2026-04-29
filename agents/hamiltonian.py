# agents/hamiltonian.py
import numpy as np


class HamiltonianAgent:
    """
    Агент, следующий по правильному замкнутому Гамильтонову циклу.
    Использует паттерн "Гребёнка" (Comb pattern), чтобы избежать телепортации.
    """

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.path = self._generate_hamiltonian_cycle()
        # Словарь для быстрого поиска индекса клетки: (row, col) -> index
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.path)}

    def _generate_hamiltonian_cycle(self):
        """
        Генерирует замкнутый путь.
        Логика:
        1. Зигзаг по колонкам 1..N-1
        2. Спуск в самый низ левой колонки (0)
        3. Подъём по колонке 0 до верха
        4. Шаг вправо в начало (0,1)
        """
        path = []

        # 1. Зигзаг по основным колонкам (1 -> grid_size-1)
        for r in range(self.grid_size):
            if r % 2 == 0:
                # Чётный ряд: слева направо (от 1 до конца)
                for c in range(1, self.grid_size):
                    path.append((r, c))
            else:
                # Нечётный ряд: справа налево (от конца до 1)
                for c in range(self.grid_size - 1, 0, -1):
                    path.append((r, c))

        # Сейчас мы в (grid_size-1, 1) - почти внизу слева

        # 2. Шаг влево в колонку 0 (самый низ)
        path.append((self.grid_size - 1, 0))

        # 3. Подъём вверх по колонке 0
        for r in range(self.grid_size - 2, -1, -1):
            path.append((r, 0))

        # 4. Теперь мы в (0,0). Замыкаем цикл шагом в (0,1)
        # (Этот шаг технически не нужен в списке, так как цикл зацикливается сам,
        # но для полноты картины можно добавить, если нужно.
        # Главное, что (0,0) является концом списка, а (0,1) - началом).

        return path

    def get_action(self, env):
        """
        Возвращает действие для движения к следующей клетке в цикле.
        """
        head = env.snake[0]

        # Находим, где мы сейчас в цикле
        if head not in self.pos_to_idx:
            # На всякий случай, если змейка вылетела (не должно быть)
            return env.direction

        head_idx = self.pos_to_idx[head]

        # Следующая позиция (с зацикливанием)
        next_idx = (head_idx + 1) % len(self.path)
        next_pos = self.path[next_idx]

        # Вычисляем вектор движения
        dr = next_pos[0] - head[0]
        dc = next_pos[1] - head[1]

        if dr == -1: return 0  # UP
        if dr == 1:  return 1  # DOWN
        if dc == -1: return 2  # LEFT
        if dc == 1:  return 3  # RIGHT

        return env.direction