import pygame
import time
from snake_env import SnakeEnv
from agents.voln_hamiltonian import ShortenedHamiltonianAgent  # ✅ ПРЯМОЙ ИМПОРТ!

GRID_SIZE = 10
CELL_SIZE = 30


class TestHelper:
    def __init__(self, grid_size):
        from agents.hamiltonian import HamiltonianAgent
        self.base_agent = HamiltonianAgent(grid_size=grid_size)
        self.cycle = self.base_agent.path
        self.pos_to_idx = self.base_agent.pos_to_idx

    def get_cycle_len(self):
        return len(self.cycle)

    def align_snake(self, env):
        head = env.snake[0]
        if head not in self.pos_to_idx:
            head = self.cycle[0]

        head_idx = self.pos_to_idx[head]
        env.snake = [head]
        for i in range(1, 3):
            prev_idx = (head_idx - i) % len(self.cycle)
            env.snake.append(self.cycle[prev_idx])

        next_idx = (head_idx + 1) % len(self.cycle)
        next_pos = self.cycle[next_idx]
        dr, dc = next_pos[0] - head[0], next_pos[1] - head[1]
        env.direction = 0 if dr == -1 else 1 if dr == 1 else 2 if dc == -1 else 3


def safe_get_cycle_pos(agent):
    return getattr(agent, 'cycle_pos', 0) % 100


def run_test(grid_size: int):
    print(f"\n{'=' * 60}")
    print(f"🧪 ShortenedHamiltonianAgent | {grid_size}x{grid_size}")
    print(f"{'=' * 60}")

    env = SnakeEnv(grid_size=grid_size, cell_size=CELL_SIZE, render_mode='human')
    test_helper = TestHelper(grid_size)
    agent = ShortenedHamiltonianAgent(grid_size)

    state = env.reset()
    test_helper.align_snake(env)

    head = env.snake[0]
    print(f"🟢 Старт | {head} | Длина: {len(env.snake)} | "
          f"Направление: {env.direction} | Цикл: {test_helper.get_cycle_len()}")

    steps = 0
    max_steps = grid_size * grid_size * 4
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    test_helper.align_snake(env)
                    steps = 0
                    print(f"🔄 Рестарт | {env.snake[0]}")
                elif event.key == pygame.K_t:
                    env.close()
                    return True
                elif event.key == pygame.K_q:
                    env.close()
                    return False

        try:
            action = agent.get_action(env)
        except Exception as e:
            print(f"💥 Ошибка: {e}")
            action = 0

        state, reward, done, info = env.step(action)
        steps += 1

        if steps % 30 == 0:
            cycle_pos = safe_get_cycle_pos(agent)
            print(f"📊 {steps:4d} | Длина:{len(env.snake):3d} | "
                  f"Еда:{env.food} | Действие:{action} | Цикл:{cycle_pos:3d}")

        if done:
            field_size = grid_size ** 2
            efficiency = len(env.snake) / field_size * 100

            status = "🏆 ПОБЕДА!" if len(env.snake) >= field_size else "💀 Смерть!"
            print(f"{status} | {efficiency:.1f}% ({len(env.snake)}/{field_size})")
            print(f"   Шагов: {steps}")
            time.sleep(2)
            running = False

    env.close()
    return True


def main():
    pygame.init()
    print("🐍 ShortenedHamiltonianAgent ТЕСТЕР")
    print("R=рестарт | T=следующий | Q=выход")
    print("🔍 Ищите DEBUG:True и скачки Цикл!")
    print("=" * 50)

    run_test(10)
    pygame.quit()


if __name__ == "__main__":
    main()