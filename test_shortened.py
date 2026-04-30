# test_shortened.py
import pygame
from snake_env import SnakeEnv
from agents.shortened_hamiltonian import ShortenedHamiltonianAgent

GRID_SIZE = 20
CELL_SIZE = 30

env = SnakeEnv(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_mode='human')
agent = ShortenedHamiltonianAgent(grid_size=GRID_SIZE)
state = env.reset()

# Выравнивание старта по циклу (как в чистом варианте)
head = env.snake[0]
head_idx = agent.pos_to_idx[head]
env.snake = [head,
             agent.path[(head_idx - 1) % len(agent.path)],
             agent.path[(head_idx - 2) % len(agent.path)]]
next_pos = agent.path[(head_idx + 1) % len(agent.path)]
dr, dc = next_pos[0] - head[0], next_pos[1] - head[1]
env.direction = 0 if dr==-1 else (1 if dr==1 else (2 if dc==-1 else 3))

print(f"🟢 Shortened Hamiltonian | Поле: {GRID_SIZE}x{GRID_SIZE}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                state = env.reset()
                head = env.snake[0]
                head_idx = agent.pos_to_idx[head]
                env.snake = [head, agent.path[(head_idx-1)%len(agent.path)], agent.path[(head_idx-2)%len(agent.path)]]
                np = agent.path[(head_idx+1)%len(agent.path)]
                dr, dc = np[0]-head[0], np[1]-head[1]
                env.direction = 0 if dr==-1 else (1 if dr==1 else (2 if dc==-1 else 3))

    action = agent.get_action(env)
    state, reward, done, info = env.step(action)

    if done:
        is_full = len(env.snake) >= GRID_SIZE ** 2
        msg = "🏆 ПОБЕДА (поле заполнено)" if is_full else ("⏱️ Таймаут" if env.steps >= env.max_steps else "💀 Смерть")
        print(f"{msg} | Score: {env.score} | Steps: {env.steps}")
        running = False

env.close()