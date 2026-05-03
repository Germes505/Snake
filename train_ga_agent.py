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

# ==================== КОНФИГУРАЦИЯ ====================
GRID_SIZE = 20
GENERATIONS = 100  # Количество поколений
POP_SIZE = 100  # Особей в популяции
ELITE_COUNT = 5  # Лучших сохраняем без изменений
MUTATION_RATE = 0.1  # Вероятность мутации гена
MUTATION_SIGMA = 0.1  # Сила мутации
CROSSOVER_PROB = 0.5  # Вероятность скрещивания
TOURNAMENT_K = 3  # Размер турнира
MAX_STEPS = 20000  # Лимит шагов за игру
SAVE_PATH = "ga_best_model.pkl"


# ==================== ОБУЧЕНИЕ ====================
def train():
    print(f"🧬 Запуск GA: {GENERATIONS} поколений, популяция {POP_SIZE}")
    start_time = time.time()

    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=20, render_mode=None)
    agent = GAAgent(
        pop_size=POP_SIZE, elite_count=ELITE_COUNT,
        mutation_rate=MUTATION_RATE, mutation_sigma=MUTATION_SIGMA,
        crossover_prob=CROSSOVER_PROB, tournament_k=TOURNAMENT_K
    )
    agent.initialize_population()

    best_scores = []
    avg_scores = []

    for gen in range(1, GENERATIONS + 1):
        fitness = []
        for genome in agent.population:
            score = agent.evaluate_individual(genome, env, max_steps=MAX_STEPS)
            fitness.append(score)

        agent.evolve(fitness)

        current_best = agent.fitness_history[-1]['best']
        current_avg = agent.fitness_history[-1]['avg']
        best_scores.append(current_best)
        avg_scores.append(current_avg)

        elapsed = time.time() - start_time
        print(f"⏱️ {elapsed / 60:.1f}м | Gen: {gen:3d} | Best: {int(current_best):3d} | "
              f"Avg: {current_avg:5.2f} | Hist Best: {int(agent.best_fitness)}")

    print(f"\n🏆 Лучший результат за всё время: {agent.best_fitness}")
    agent.save(SAVE_PATH)
    plot_results(best_scores, avg_scores)
    return agent


# ==================== ВИЗУАЛИЗАЦИЯ ====================
def plot_results(best, avg):
    plt.figure(figsize=(10, 5))
    plt.plot(best, label='Best Fitness', color='red', linewidth=2)
    plt.plot(avg, label='Average Fitness', color='blue', linestyle='--')
    plt.title('GA Training Progress (Neuroevolution)')
    plt.xlabel('Generation')
    plt.ylabel('Score (Length - 3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ga_progress.png', dpi=200, bbox_inches='tight')
    print("📊 Графики сохранены в 'ga_progress.png'")
    plt.show()


# ==================== ЗАПУСК МОДЕЛИ ====================
def play_best_model(path=SAVE_PATH):
    agent = GAAgent()
    agent.load("ga_best_model_1.pkl")
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=30, render_mode='human')
    env.reset()
    state = agent._get_state(env)

    # Создаём сеть для запуска
    net = GANet()
    net.set_weights(agent.best_genome)

    print("🐍 Запуск лучшей GA-особи. R=рестарт, Esc=выход")
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
                    state = agent._get_state(env)

        action = net.get_action(state, env.direction)
        _, _, done, info = env.step(action)
        state = agent._get_state(env)

        if done:
            print(f"💀 Game Over. Score: {info['length'] - 3}")
            env.reset()
            state = agent._get_state(env)
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