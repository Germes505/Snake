# agents/q_agent.py
import random
import numpy as np
from collections import deque


class QAgent:
    """
    Агент, обучаемый методом Q-Learning (Tabular RL).
    Использует Feature Engineering для представления состояния.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        # Q-таблица: Ключ = Tuple(state), Значение = List[Q-values for 4 actions]
        self.q_table = {}

        # Гиперпараметры
        self.alpha = alpha  # Learning Rate (скорость обучения)
        self.gamma = gamma  # Discount Factor (важность будущих наград)
        self.epsilon = epsilon  # Exploration Rate (вероятность случайного хода)

        # Действия: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.actions = [0, 1, 2, 3]

    def get_state(self, env):
        """
        Преобразует сложное состояние среды в компактный вектор признаков (Tuple).
        Относительные координаты:
        - Danger: [Прямо, Справа, Слева]
        - Food: [Прямо, Справа, Слева]
        - Direction: [Влево, Вправо] (текущее направление кодируется неявно через другие)
        """
        head = env.snake[0]

        # Определяем точки вокруг головы (clockwise: straight, right, left)
        # Векторы направлений: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        # dx, dy
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        current_dir = dirs[env.direction]

        # Вычисляем векторы для Straight, Right, Left
        # Если текущее UP (-1, 0): Straight=(-1,0), Right=(0,1), Left=(0,-1)
        # Если текущее DOWN (1, 0): Straight=(1,0), Right=(0,-1), Left=(0,1)
        # И так далее...

        point_straight = (head[0] + current_dir[0], head[1] + current_dir[1])
        point_right = (head[0] - current_dir[1], head[1] + current_dir[0])  # Поворот на 90
        point_left = (head[0] + current_dir[1], head[1] - current_dir[0])  # Поворот на -90

        # 1. Danger (Препятствия: Стены или Тело)
        danger_straight = self._is_collision(point_straight, env)
        danger_right = self._is_collision(point_right, env)
        danger_left = self._is_collision(point_left, env)

        # 2. Direction (Куда смотрим сейчас? Вектор движения)
        # Кодируем как 4 бита: [UP, DOWN, LEFT, RIGHT]
        dir_up = 1 if env.direction == 0 else 0
        dir_down = 1 if env.direction == 1 else 0
        dir_left = 1 if env.direction == 2 else 0
        dir_right = 1 if env.direction == 3 else 0

        # 3. Food (Где еда относительно направления движения?)
        # True, если еда в этом направлении
        food_straight = (env.food[0] == point_straight[0] and env.food[1] == point_straight[1]) or \
                        (env.food[0] - head[0]) * current_dir[0] + (env.food[1] - head[1]) * current_dir[
                            1] > 0  # Упрощенно: проекция

        # Более точная проверка для Food:
        # Еда Прямо: если вектор к еде сонаправлен с движением
        food_straight = (env.food[0] - head[0]) * current_dir[0] + (env.food[1] - head[1]) * current_dir[1] > 0

        # Еда Справа: если векторное произведение (2D analog) положительное
        food_right = (env.food[0] - head[0]) * (-current_dir[1]) + (env.food[1] - head[1]) * current_dir[0] > 0

        # Еда Слева: отрицательное
        food_left = (env.food[0] - head[0]) * current_dir[1] + (env.food[1] - head[1]) * (-current_dir[0]) > 0

        state = (
            danger_straight, danger_right, danger_left,
            dir_up, dir_down, dir_left, dir_right,
            food_straight, food_right, food_left
        )
        return state

    def _is_collision(self, point, env):
        """Проверяет, является ли точка стеной или частью тела змейки."""
        r, c = point
        # Проверка стен
        if not (0 <= r < env.grid_size and 0 <= c < env.grid_size):
            return True
        # Проверка тела
        if point in env.snake:
            return True
        return False

    def get_action(self, state):
        self._ensure_state_exists(state)

        # Бонус за исследование редко посещаемых состояний
        visit_count = len([v for v in self.q_table.values() if any(v)])
        curiosity_bonus = 0.1 / (1 + visit_count / 1000)

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [q + curiosity_bonus for q in self.q_table[state]]
            max_val = max(q_values)
            max_indices = [i for i, v in enumerate(q_values) if v == max_val]
            return random.choice(max_indices)

    def train(self, state, action, reward, next_state, done):
        """Обновляет Q-таблицу по формуле Bellman."""
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        # Текущая оценка
        old_value = self.q_table[state][action]

        # Максимальная будущая награда (если игра не окончена)
        max_future_q = max(self.q_table[next_state]) if not done else 0

        # Формула Q-Learning
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * max_future_q)

        self.q_table[state][action] = new_value

    def _ensure_state_exists(self, state):
        """Инициализирует состояние в таблице, если его там нет."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)