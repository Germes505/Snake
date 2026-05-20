# test_hamiltonian.py
import time

import pygame
from snake_env import SnakeEnv
from agents.hamiltonian import HamiltonianAgent

GRID_SIZE = 20 # Можно менять на 10
CELL_SIZE = 30

env = SnakeEnv(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_mode='human')
agent = HamiltonianAgent(grid_size=GRID_SIZE)
state = env.reset()

# 🔑 КРИТИЧЕСКИ ВАЖНО: выравниваем тело змейки "позади" головы по циклу
head = env.snake[0]
head_idx = agent.pos_to_idx[head]
env.snake = [head,
             agent.path[(head_idx - 1) % len(agent.path)],
             agent.path[(head_idx - 2) % len(agent.path)]]

# Вычисляем стартовое направление (вперёд по циклу)
next_pos = agent.path[(head_idx + 1) % len(agent.path)]
dr, dc = next_pos[0] - head[0], next_pos[1] - head[1]
if dr == -1: env.direction = 0
elif dr == 1: env.direction = 1
elif dc == -1: env.direction = 2
else: env.direction = 3

print(f"🟢 Запуск | Поле: {GRID_SIZE}x{GRID_SIZE} | Стартовая позиция: {head}")

running = True
scor_ = 0
step_ = 0
n = 0
while running:
    # Обработка событий (обязательно каждый кадр, иначе "Не отвечает")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                state = env.reset()
                head = env.snake[0]
                head_idx = agent.pos_to_idx[head]
                env.snake = [head,
                             agent.path[(head_idx - 1) % len(agent.path)],
                             agent.path[(head_idx - 2) % len(agent.path)]]
                next_pos = agent.path[(head_idx + 1) % len(agent.path)]
                dr, dc = next_pos[0] - head[0], next_pos[1] - head[1]
                if dr == -1: env.direction = 0
                elif dr == 1: env.direction = 1
                elif dc == -1: env.direction = 2
                else: env.direction = 3

    action = agent.get_action(env)
    state, reward, done, info = env.step(action)

    if done:
        n += 1
        scor_ += info['length']
        step_ += env.steps
        if len(env.snake) >= GRID_SIZE ** 2:
            print(f"🏆 ПОБЕДА! Поле полностью заполнено. Score: {env.score}, Steps: {env.steps}")
        elif env.steps >= env.max_steps:
            print(f"⏱️ Таймаут. Score: {env.score}")
        else:
            print(f"💀 Смерть. Score: {env.score}")
        running = True  # Останавливаем цикл после завершения
        time.sleep(10)

env.close()