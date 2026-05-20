# test_greedy.py
"""Тестовый запуск для GreedyAgent."""

import pygame
from snake_env import SnakeEnv
from agents.greedy import GreedyAgent

# === Настройки ===
GRID_SIZE = 8
CELL_SIZE = 30
RENDER_MODE = 'human'  # 'human' для отрисовки, 'none' для быстрых тестов


def main():
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_mode=RENDER_MODE)
    agent = GreedyAgent(grid_size=GRID_SIZE)
    state = env.reset()

    print(f"🟢 GreedyAgent запущен | Поле: {GRID_SIZE}x{GRID_SIZE}")
    print("Управление: [R] рестарт, [ESC/Q] выход")

    running = True
    stats = {"episodes": 0, "total_score": 0, "total_steps": 0, "wins": 0}

    while running:
        # 🎮 Обработка событий Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    state = env.reset()
                    print(f"🔄 Эпизод #{stats['episodes'] + 1} начат")

        # 🤖 Получаем действие от агента
        action = agent.get_action(env)
        state, reward, done, info = env.step(action)

        # 📊 Обработка конца эпизода
        if done:
            stats["episodes"] += 1
            stats["total_score"] += info['length']
            stats["total_steps"] += env.steps

            # Определение результата
            if len(env.snake) >= GRID_SIZE ** 2:
                result = "🏆 ПОБЕДА!"
                stats["wins"] += 1
            elif env.steps >= env.max_steps:
                result = "⏱️ Таймаут"
            else:
                result = "💀 Смерть"

            print(f"{result} | Score: {env.score:3d} | "
                  f"Steps: {env.steps:4d} | Length: {len(env.snake):3d}")

            # Авторм рестарт для демо
            state = env.reset()

    # 📈 Итоговая статистика
    env.close()
    if stats["episodes"] > 0:
        avg_len = stats["total_score"] / stats["episodes"]
        avg_steps = stats["total_steps"] / stats["episodes"]
        win_rate = stats["wins"] / stats["episodes"] * 100
        print(f"\n📊 Статистика ({stats['episodes']} эпизодов):")
        print(f"   Победы: {stats['wins']} ({win_rate:.1f}%)")
        print(f"   Ср. длина змейки: {avg_len:.1f} / {GRID_SIZE ** 2}")
        print(f"   Ср. шагов на эпизод: {avg_steps:.1f}")


if __name__ == "__main__":
    main()