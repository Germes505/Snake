# test_hamiltonian.py
import pygame
from snake_env import SnakeEnv
from agents.hamiltonian import HamiltonianAgent

GRID_SIZE = 10
CELL_SIZE = 50

env = SnakeEnv(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_mode='human')
agent = HamiltonianAgent(grid_size=GRID_SIZE)
state = env.reset()

# --- ИСПРАВЛЕНИЕ: Выравниваем змейку строго по потоку цикла ---
head = env.snake[0]
head_idx = agent.pos_to_idx[head]

# Тело ставим туда, где голова была 1 и 2 шага назад
tail1 = agent.path[(head_idx - 1) % len(agent.path)]
tail2 = agent.path[(head_idx - 2) % len(agent.path)]
env.snake = [head, tail1, tail2]

# Вычисляем корректное стартовое направление (вперёд по циклу)
next_pos = agent.path[(head_idx + 1) % len(agent.path)]
dr = next_pos[0] - head[0]
dc = next_pos[1] - head[1]

if dr == -1: env.direction = 0  # UP
elif dr == 1: env.direction = 1 # DOWN
elif dc == -1: env.direction = 2 # LEFT
elif dc == 1: env.direction = 3 # RIGHT
# ---------------------------------------------------------------

print(f"🟢 Гамильтонов цикл | Поле: {GRID_SIZE}x{GRID_SIZE}")
print(f"📍 Старт: голова {head} | Направление: {env.direction}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                state = env.reset()
                # При рестарте снова выравниваем змейку
                head = env.snake[0]
                head_idx = agent.pos_to_idx[head]
                env.snake = [head,
                             agent.path[(head_idx - 1) % len(agent.path)],
                             agent.path[(head_idx - 2) % len(agent.path)]]
                next_pos = agent.path[(head_idx + 1) % len(agent.path)]
                dr = next_pos[0] - head[0]
                dc = next_pos[1] - head[1]
                if dr == -1: env.direction = 0
                elif dr == 1: env.direction = 1
                elif dc == -1: env.direction = 2
                elif dc == 1: env.direction = 3

    action = agent.get_action(env)
    state, reward, done, info = env.step(action)

    if done:
        if env.steps >= env.max_steps:
            print(f"⏱️ Лимит шагов! Змейка выжила. Score: {env.score}")
        else:
            print(f"💀 Смерть! Score: {env.score} | Steps: {env.steps}")
        state = env.reset()
        # Повторное выравнивание после смерти
        head = env.snake[0]
        head_idx = agent.pos_to_idx[head]
        env.snake = [head,
                     agent.path[(head_idx - 1) % len(agent.path)],
                     agent.path[(head_idx - 2) % len(agent.path)]]
        next_pos = agent.path[(head_idx + 1) % len(agent.path)]
        dr = next_pos[0] - head[0]
        dc = next_pos[1] - head[1]
        if dr == -1: env.direction = 0
        elif dr == 1: env.direction = 1
        elif dc == -1: env.direction = 2
        elif dc == 1: env.direction = 3

env.close()