# test_play.py
import pygame
from snake_env import SnakeEnv

env = SnakeEnv(grid_size=20, cell_size=30, render_mode='human')
state = env.reset()

running = True
current_action = SnakeEnv.RIGHT

while running:
    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Проверяем все возможные клавиши
            if event.key in (pygame.K_w, pygame.K_UP):
                new_action = SnakeEnv.UP
            elif event.key in (pygame.K_s, pygame.K_DOWN):
                new_action = SnakeEnv.DOWN
            elif event.key in (pygame.K_a, pygame.K_LEFT):
                new_action = SnakeEnv.LEFT
            elif event.key in (pygame.K_d, pygame.K_RIGHT):
                new_action = SnakeEnv.RIGHT
            else:
                # Если нажата другая клавиша - пропускаем
                continue

            # Проверка на противоположное направление
            # UP(0) <-> DOWN(1), LEFT(2) <-> RIGHT(3)
            opposite = current_action ^ 1
            if new_action != opposite:
                current_action = new_action
            # Иначе просто игнорируем нажатие (не меняем направление)

    # Отправляем текущее направление в игру
    state, reward, done, info = env.step(current_action)

    if done:
        print(f"🎮 Game Over! Score: {env.score} | Steps: {info['steps']}")
        state = env.reset()
        current_action = SnakeEnv.RIGHT  # Сброс на право

env.close()