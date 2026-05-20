# test_play.py
import pygame
from snake_env import SnakeEnv

# Инициализация среды
env = SnakeEnv(grid_size=20, cell_size=30, render_mode='human')
state = env.reset()

# Первоначальная отрисовка стартового поля
env.render()

running = True
current_action = SnakeEnv.RIGHT
clock = pygame.time.Clock()  # Контроль FPS для экономии ресурсов CPU

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            new_action = None

            # Сопоставление клавиш с действиями
            if event.key in (pygame.K_w, pygame.K_UP):
                new_action = SnakeEnv.UP
            elif event.key in (pygame.K_s, pygame.K_DOWN):
                new_action = SnakeEnv.DOWN
            elif event.key in (pygame.K_a, pygame.K_LEFT):
                new_action = SnakeEnv.LEFT
            elif event.key in (pygame.K_d, pygame.K_RIGHT):
                new_action = SnakeEnv.RIGHT

            # Обрабатываем только валидные направления
            if new_action is not None:
                # Запрет поворота на 180° (UP↔DOWN, LEFT↔RIGHT)
                if new_action != (current_action ^ 1):
                    current_action = new_action

                    # 🐍 Шаг выполняется СТРОГО по нажатию клавиши
                    state, reward, done, info = env.step(current_action)

                    # Обработка завершения игры
                    if done:
                        print(f" Game Over! Score: {env.score} | Steps: {info['steps']}")
                        state = env.reset()
                        current_action = SnakeEnv.RIGHT
                        env.render()  # Отрисовка нового эпизода

    # Ограничение частоты обновления цикла (60 FPS)
    # Без этой строки скрипт будет потреблять ~100% одного ядра CPU
    clock.tick(60)

env.close()