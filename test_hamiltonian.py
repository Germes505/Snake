# test_hamiltonian.py
import pygame
from snake_env import SnakeEnv
from agents.hamiltonian import HamiltonianAgent

env = SnakeEnv(grid_size=20, cell_size=30, render_mode='human')
agent = HamiltonianAgent(grid_size=20)
state = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                state = env.reset()
                print(f"Restarted! Score: {env.score}")

    # Агент принимает решение
    action = agent.get_action(env)
    state, reward, done, info = env.step(action)

    if done:
        print(f"🎮 Game Over! Score: {env.score} | Steps: {info['steps']}")
        state = env.reset()

env.close()