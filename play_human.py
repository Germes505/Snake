# play_human.py
import pygame
import sys
from snake_env import SnakeEnv

GRID_SIZE = 20
CELL_SIZE = 30

def main():
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_mode='human')
    env.reset()
    print("🐍 Управление: ← ↑ ↓ → | R = рестарт | Esc = выход")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    print("🔄 Рестарт")
                elif event.key == pygame.K_UP:
                    env.step(SnakeEnv.UP)
                elif event.key == pygame.K_DOWN:
                    env.step(SnakeEnv.DOWN)
                elif event.key == pygame.K_LEFT:
                    env.step(SnakeEnv.LEFT)
                elif event.key == pygame.K_RIGHT:
                    env.step(SnakeEnv.RIGHT)
        pygame.display.flip()
        env.clock.tick(15)  # Скорость игры (кадры в секунду)

    env.close()

if __name__ == "__main__":
    main()