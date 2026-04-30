# test_shortened.py
import pygame
import time
from snake_env import SnakeEnv
from agents.shortened_hamiltonian import ShortenedHamiltonianAgent

GRID_SIZE = 20
CELL_SIZE = 30

env = SnakeEnv(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_mode='human')
agent = ShortenedHamiltonianAgent(grid_size=GRID_SIZE, debug=True)
state = env.reset()


def align_snake():
    head = env.snake[0]
    idx = agent.pos_to_idx[head]
    body_1 = agent.path[(idx - 1) % len(agent.path)]
    body_2 = agent.path[(idx - 2) % len(agent.path)]
    env.snake = [head, body_1, body_2]

    next_pos = agent.path[(idx + 1) % len(agent.path)]
    dr, dc = next_pos[0] - head[0], next_pos[1] - head[1]
    env.direction = 0 if dr == -1 else (1 if dr == 1 else (2 if dc == -1 else 3))


align_snake()
print(f" Wave BFS Agent | {GRID_SIZE}x{GRID_SIZE} | Старт: {env.snake[0]} | Dir: {env.direction}")
time.sleep(2)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("\n🔄 Рестарт")
                state = env.reset()
                align_snake()
                print(f"   Новое тело: {env.snake} | Dir: {env.direction}")

    action = agent.get_action(env)
    state, reward, done, info = env.step(action)

    if done:
        is_full = len(env.snake) >= GRID_SIZE ** 2
        if is_full:
            print(f"\n ПОБЕДА! Score: {env.score}")
        elif env.steps >= env.max_steps:
            print(f"\n⏱️ Таймаут. Score: {env.score}")
        else:
            print(f"\n💀 Смерть. Score: {env.score} | Steps: {env.steps}")

        print("⏸️ Пауза 8 сек (скриншот)...")
        time.sleep(8)
        running = False

env.close()