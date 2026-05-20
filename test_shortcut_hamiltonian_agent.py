# В main.py или test.py
from snake_env import SnakeEnv
from agents.shortcut_hamiltonian_agent import HamiltonianSnakeSolver

grid_size = 20  # Размер поля
env = SnakeEnv(grid_size=grid_size, render_mode='human')
solver = HamiltonianSnakeSolver(grid_size)

state = env.reset()
total_reward = 0
print("Запуск Hamiltonian Snake Solver...")
print("Гарантированная выживаемость + оптимизация!")

try:
    while not env.done:
        action = solver.get_next_direction(env)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"\n=== ЭПИЗОД ЗАВЕРШЕН ===")
            print(f"Счёт: {env.score}")
            print(f"Длина змейки: {info['length']}")
            print(f"Шаги: {info['steps']}")
            print(f"Награда: {total_reward:.1f}")

            if env.food is None:
                print("🎉 ПОБЕДА! Поле заполнено!")
            else:
                print("Игра окончена (timeout или заполнение)")

            restart = input("\nНовая игра? (y/n): ").lower()
            if restart == 'y':
                state = env.reset()
                total_reward = 0
                solver = HamiltonianSnakeSolver(grid_size)
            else:
                break
except KeyboardInterrupt:
    print("\nОстановка по Ctrl+C")

env.close()