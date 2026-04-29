# snake_env.py
import pygame
import numpy as np
import random
import sys
# Тестовость

class SnakeEnv:
    # Коды действий
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, grid_size=10, cell_size=40, render_mode='human'):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        self.render_mode = render_mode
        self.max_steps = grid_size * grid_size * 200  # Защита от бесконечных эпизодов

        # Инициализация Pygame только для режима отрисовки
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake RL Environment")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        # Змейка стартует с длиной 3 в центре
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = self.RIGHT
        self.score = 0
        self.steps = 0
        self.done = False
        self._place_food()
        return self.get_state()

    def step(self, action):
        if self.done:
            raise ValueError("Episode finished. Call reset() first.")

        # Запрет разворота на 180°
        opposite = self.direction ^ 1  # UP(0)↔DOWN(1), LEFT(2)↔RIGHT(3)
        if action != opposite:
            self.direction = action

        head_r, head_c = self.snake[0]
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][self.direction]
        new_head = (head_r + dr, head_c + dc)

        reward = -0.1  # Штраф за шаг (стимулирует быстрее есть)
        self.steps += 1

        # Проверка столкновения со стеной
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.done = True
            reward = -10
        # Проверка столкновения с собой
        elif new_head in self.snake:
            self.done = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()

        # Лимит шагов (важно для стабильного RL)
        if self.steps >= self.max_steps:
            self.done = True

        if self.render_mode == 'human':
            self.render()

        info = {"steps": self.steps, "length": len(self.snake)}
        return self.get_state(), reward, self.done, info

    def _place_food(self):
        while True:
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if (r, c) not in self.snake:
                self.food = (r, c)
                break

    def get_state(self):
        """
        Возвращает наблюдение. Сейчас это полная сетка (удобно для отладки и DQN позже).
        Для табличного Q-Learning в следующем шаге добавим метод get_features().
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for r, c in self.snake:
            grid[r, c] = 1
        grid[self.food] = 2
        return grid

    def render(self):
        if self.render_mode != 'human':
            return

        # Чёрный фон
        self.screen.fill((0, 0, 0))

        # Рисуем еду - красный квадрат
        pygame.draw.rect(self.screen, (255, 50, 50),
                         (self.food[1] * self.cell_size,
                          self.food[0] * self.cell_size,
                          self.cell_size, self.cell_size))

        # Рисуем змейку - зелёные блоки
        for i, (r, c) in enumerate(self.snake):
            # Голова - ярче, тело - темнее
            if i == 0:
                color = (100, 255, 100)  # Голова - светло-зелёная
            else:
                color = (0, 200, 0)  # Тело - зелёное

            pygame.draw.rect(self.screen, color,
                             (c * self.cell_size,
                              r * self.cell_size,
                              self.cell_size - 1, self.cell_size - 1))

        # Минимальный HUD (опционально - можно закомментировать)
        font = pygame.font.SysFont("consolas", 14)
        txt = font.render(f"Score: {self.score}", True, (200, 200, 200))
        self.screen.blit(txt, (10, self.window_size - 25))

        pygame.display.flip()
        self.clock.tick(120)  # 60 FPS для плавности

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
