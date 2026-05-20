# train_ga_agent.py
"""
Тестер и скрипт обучения GA-агента.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pygame
from snake_env import SnakeEnv
from agents.ga_agent import GAAgent, GANet
import os

# ==================== КОНФИГУРАЦИЯ ====================
GRID_SIZE = 20
GENERATIONS = 7000  # Количество поколений
POP_SIZE = 90  # Особей в популяции
ELITE_COUNT = 8  # Лучших сохраняем без изменений
MUTATION_RATE = 0.1  # Вероятность мутации гена
MUTATION_SIGMA = 0.15  # Сила мутации
CROSSOVER_PROB = 0.7  # Вероятность скрещивания
TOURNAMENT_K = 3  # Размер турнира
MAX_STEPS = 400  # Лимит шагов за игру
N_ISLANDS = 3 # Количество островов
SAVE_PATH = "ga_best_model_ostrova"


# ==================== ОБУЧЕНИЕ ====================
def train():
    print(f"🧬 Island GA: {GENERATIONS} gen, {POP_SIZE} pop, {N_ISLANDS} islands")
    start_time = time.time()

    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=20, render_mode=None)
    agent = GAAgent(
        pop_size=POP_SIZE, elite_count=ELITE_COUNT,
        mutation_rate=MUTATION_RATE, mutation_sigma=MUTATION_SIGMA,
        crossover_prob=CROSSOVER_PROB, tournament_k=TOURNAMENT_K, hidden_size=48
    )
    agent.initialize_population()

    best_scores = []
    avg_scores = []
    avg_steps_per_gen = []

    for gen in range(1, GENERATIONS + 1):
        # Оценка всех островов
        fitness_by_island = []
        steps_by_island = []
        for island in agent.islands:
            results = [agent.evaluate_individual(g, env) for g in island]
            fitness = [r[0] for r in results]
            steps = [r[1] for r in results]
            fitness_by_island.append(fitness)
            steps_by_island.append(steps)

        agent.evolve(fitness_by_island)

        current_best = agent.fitness_history[-1]['best']
        current_avg = agent.fitness_history[-1]['avg']
        best_scores.append(current_best)
        avg_scores.append(current_avg)

        all_steps = [s for island in steps_by_island for s in island]
        avg_steps = np.mean(all_steps) if all_steps else 0
        avg_steps_per_gen.append(avg_steps)

        if gen % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"⏱️ {elapsed:.1f}м | Gen: {gen:4d} | Best: {int(current_best):3d} | "
                  f"Avg: {current_avg:5.2f}  | Steps: {avg_steps:6.1f} | Hist Best: {int(agent.best_fitness)}")

    print(f"\n🏆 Лучший результат за всё время: {int(agent.best_fitness)}")
    agent.save(f'{SAVE_PATH}.pkl')
    plot_results(best_scores, avg_scores, avg_steps_per_gen)
    return agent


# ==================== ВИЗУАЛИЗАЦИЯ ====================
# ==================== ВИЗУАЛИЗАЦИЯ ====================
def plot_results(best, avg, steps=None):
    """
    Визуализация прогресса обучения.

    Параметры:
    ----------
    best : list
        Лучшие значения фитнеса по поколениям.
    avg : list
        Средние значения фитнеса по поколениям.
    steps : list, optional
        Среднее количество шагов по поколениям.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 📈 Верхний график: фитнес
    ax1.plot(best, label='Best Fitness', color='red', linewidth=2, alpha=0.9)
    ax1.plot(avg, label='Average Fitness', color='blue', linestyle='--', linewidth=2)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('GA Training Progress: Fitness Dynamics', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_axisbelow(True)

    # 👣 Нижний график: шаги (если переданы)
    if steps is not None:
        ax2.plot(steps, color='green', linewidth=1.5, label='Avg Steps per Episode')
        ax2.set_xlabel('Generation', fontsize=10)
        ax2.set_ylabel('Steps', fontsize=10)
        ax2.set_title('Episode Duration Dynamics', fontsize=11, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.set_axisbelow(True)

    # ✅ Общие настройки
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Графики сохранены в {SAVE_PATH}(300 DPI)")
    plt.show()


# ==================== ЗАПУСК МОДЕЛИ ====================
def play_best_model(path=f"{SAVE_PATH}.pkl"):
    # 🔍 Определяем архитектуру модели
    if "1.pkl" in path or "old" in path:
        hidden_size = 24
        print("⚠️  Загрузка СТАРОЙ модели (24 нейрона)")
    else:
        hidden_size = 48
        print("✅ Загрузка НОВОЙ модели (48 нейронов)")

    agent = GAAgent(
        pop_size=100, elite_count=5, mutation_rate=0.1,
        mutation_sigma=0.1, crossover_prob=0.5,
        tournament_k=3, hidden_size=hidden_size
    )
    agent.load(path)

    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=30, render_mode='human')
    env.reset()
    state = agent._get_state(env)

    net = GANet(input_size=12, hidden_size=hidden_size, output_size=4)
    net.set_weights(agent.best_genome)

    # 🔥 Статистика для вывода
    episode = 0
    total_length = 0
    total_steps = 0
    window = 10  # Окно для скользящего среднего

    # 🔥 Параметры динамического лимита шагов
    base_steps = 400
    step_bonus = 400
    step_budget = base_steps
    steps_taken = 0

    print("\n🎮 Управление: R = рестарт, Esc = выход")
    print(f"📊 Статистика: [Эпизод] Длина | Шаги | Ср. длина (10) | Ср. шаги (10)")
    print("-" * 70)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Принудительный рестарт с выводом статистики
                    episode += 1
                    length = len(env.snake)
                    print(f"#{episode:4d} | {length:3d} | {steps_taken:5d} | "
                          f"{total_length / max(1, episode):5.1f} | {total_steps / max(1, episode):6.1f}")
                    env.reset()
                    state = agent._get_state(env)
                    step_budget = base_steps
                    steps_taken = 0

        # 🔥 Проверка лимита шагов
        if steps_taken >= step_budget:
            episode += 1
            length = len(env.snake)
            total_length += length
            total_steps += steps_taken

            # Вывод статистики
            avg_len = total_length / episode
            avg_steps = total_steps / episode
            print(f"#{episode:4d} | {length:3d} | {steps_taken:5d} | "
                  f"{avg_len:5.1f} | {avg_steps:6.1f}")

            # Рестарт
            env.reset()
            state = agent._get_state(env)
            step_budget = base_steps
            steps_taken = 0
            continue

        action = net.get_action(state, env.direction)
        old_head = tuple(env.snake[0])

        _, reward, done, info = env.step(action)
        state = agent._get_state(env)
        steps_taken += 1

        # 🔥 Детекция поедания яблока
        if reward > 0 or (tuple(env.snake[0]) != old_head and len(env.snake) > info.get('length', len(env.snake))):
            step_budget += step_bonus
            # Можно раскомментировать для отладки:
            # print(f"🍎 +400 шагов (бюджет: {step_budget})")

        # 🔥 Обработка естественного завершения игры
        if done:
            episode += 1
            length = info.get('length', len(env.snake))
            total_length += length
            total_steps += steps_taken

            # Вывод статистики
            avg_len = total_length / episode
            avg_steps = total_steps / episode
            print(f"#{episode:4d} | {length:3d} | {steps_taken:5d} | "
                  f"{avg_len:5.1f} | {avg_steps:6.1f}")

            # Рестарт
            env.reset()
            state = agent._get_state(env)
            step_budget = base_steps
            steps_taken = 0

    # 🔥 Финальная статистика при выходе
    if episode > 0:
        print("\n" + "=" * 70)
        print(f"🏁 ИТОГО: Эпизодов: {episode}")
        print(f"📈 Средняя длина змейки: {total_length / episode:.2f}")
        print(f"👣 Среднее количество шагов: {total_steps / episode:.1f}")
        print(f"🎯 Лучший результат: {max(total_length / max(1, i) for i in range(1, episode + 1)):.1f}")
        print("=" * 70)

    env.close()

# ==================== ТОЧКА ВХОДА ====================
if __name__ == "__main__":
    print("1 — Обучить GA-агента")
    print("2 — Запустить лучшую сохранённую модель")
    choice = input("Выбор [1/2]: ").strip()

    if choice == '1':
        train()
    elif choice == '2':
        play_best_model()
    else:
        print("❌ Неверный выбор")