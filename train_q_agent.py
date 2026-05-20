# train_q_agent.py
import pygame
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from snake_env import SnakeEnv
from agents.q_agent import QAgent

# --- КОНФИГУРАЦИЯ ---
GRID_SIZE = 20
EPISODES = 300000  # Увеличили количество игр
BASE_STEPS = 400
STEP_BONUS = 400

# 🎯 УЛУЧШЕННЫЕ ГИПЕРПАРАМЕТРЫ
ALPHA = 0.001  # Чуть выше скорость обучения
GAMMA = 0.99  # Больше внимания будущему
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99998  # МЕДЛЕННЕЕ затухание (агент дольше исследует)

SAVE_MODEL = True
MODEL_PATH = "q_agent_model_steps_.pkl"

def smooth_curve(data, window_size=100):
    """Сглаживание кривой методом скользящего среднего"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def train():
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=20, render_mode=None)
    agent = QAgent(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START)

    score_history = []
    epsilon_history = []
    steps_history = []
    avg_score_window = deque(maxlen=100)
    avg_steps_window = deque(maxlen=100)

    # Для отслеживания лучших результатов
    best_score = 0
    best_episode = 0

    print(f"🚀 Начало обучения: {EPISODES} эпизодов...")
    print(f"📊 Гиперпараметры: α={ALPHA}, γ={GAMMA}, ε_decay={EPSILON_DECAY}")

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        agent_state = agent.get_state(env)
        total_reward = 0
        done = False
        steps_taken = 0
        step_budget = BASE_STEPS
        prev_len = len(env.snake)

        while not done:
            action = agent.get_action(agent_state)
            _, env_reward, done, info = env.step(action)
            next_state = agent.get_state(env)
            steps_taken += 1

            # 🍎 Детекция съеденной еды и расширение бюджета
            if len(env.snake) > prev_len:
                step_budget = STEP_BONUS
                prev_len = len(env.snake)

            # ️ Проверка динамического лимита
            if steps_taken >= step_budget:
                done = True

            # 🎯 Формирование награды (логика сохранена)
            reward = env_reward
            if reward == 0:
                reward = 0.001
            elif reward > 0:
                reward = 20
            elif done:
                reward = -150

            agent.train(agent_state, action, reward, next_state, done)
            agent_state = next_state
            total_reward += reward

        score = info['length']
        score_history.append(score)
        steps_history.append(steps_taken)
        epsilon_history.append(agent.epsilon)
        avg_score_window.append(score)
        avg_steps_window.append(steps_taken)

        if score > best_score:
            best_score = score
            best_episode = episode
            with open("q_agent_best.pkl", "wb") as f:
                pickle.dump(agent, f)

        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        if episode % 100 == 0:
            avg = sum(avg_score_window) / len(avg_score_window)
            avg_steps = sum(steps_history[-100:]) / 100
            print(f"Ep: {episode:5d} | Score: {score:3d} | Avg: {avg:5.2f} | "
                  f"Steps: {avg_steps:6.1f} | ε: {agent.epsilon:.4f} | Best: {best_score}")

    print(f"\n🏆 Лучший результат: {best_score} (эпизод {best_episode})")

    # --- СОХРАНЕНИЕ ---
    if SAVE_MODEL:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(agent, f)
        print(f"💾 Модель сохранена в {MODEL_PATH}")

    # --- 📊 УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    visualize_training(score_history, epsilon_history, steps_history)


def visualize_training(scores, epsilons, steps):
    """Продвинутая визуализация (исправлена длина массивов)"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Progress of Q-Learning Agent', fontsize=14, fontweight='bold')

    # 1. Сырые данные + сглаживание
    axs[0, 0].plot(scores, alpha=0.3, label='Raw', color='gray')
    smooth_100 = np.convolve(scores, np.ones(100) / 100, mode='valid')
    axs[0, 0].plot(range(100 - 1, len(scores)), smooth_100, label='Moving Avg (100)',
                   color='blue', linewidth=2)
    axs[0, 0].set_title('Score per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Epsilon decay
    axs[0, 1].plot(epsilons, color='red', linewidth=2)
    axs[0, 1].set_title('Exploration Rate (Epsilon)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Epsilon')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Steps per Episode
    axs[1, 0].plot(steps, alpha=0.4, color='green', label='Raw', linewidth=0.5)
    if len(steps) >= 100:
        smooth_steps = np.convolve(steps, np.ones(100) / 100, mode='valid')
        axs[1, 0].plot(range(99, len(steps)), smooth_steps,
                       label='Moving Avg (100)', color='darkgreen', linewidth=2)
    axs[1, 0].axhline(y=BASE_STEPS, color='orange', linestyle=':',
                      label=f'Base Steps ({BASE_STEPS})', alpha=0.7)
    axs[1, 0].set_title('Steps per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Steps')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Последние 500 эпизодов (✅ ИСПРАВЛЕНО)
    if len(scores) > 500:
        recent_scores = scores[-500:]
        window = 50
        recent_smooth = np.convolve(recent_scores, np.ones(window) / window, mode='valid')

        # X для сырых данных (500 точек)
        x_raw = range(len(scores) - 500, len(scores))
        # X для сглаженных данных (451 точка, т.к. valid-свёртка укорачивает на window-1)
        x_smooth = range(len(scores) - 500 + window - 1, len(scores))

        axs[1, 1].plot(x_raw, recent_scores, alpha=0.3, color='gray')
        axs[1, 1].plot(x_smooth, recent_smooth, color='purple', linewidth=2)
        axs[1, 1].set_title('Last 500 Episodes (Detailed)')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Score')
        axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{MODEL_PATH}.png', dpi=300, bbox_inches='tight')
    print("📊 Графики сохранены в 'training_progress.png'")
    plt.show()


def play_trained_model(model_path):
    """Визуализация обученного агента с динамическим лимитом шагов"""
    try:
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Модель '{model_path}'.pkl не найдена!")
        return

    agent.epsilon = 0.0
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=30, render_mode='human')

    BASE_STEPS = 400
    STEP_BONUS = 400

    env.reset()
    agent_state = agent.get_state(env)
    step_budget = BASE_STEPS
    steps_taken = 0
    prev_len = len(env.snake)

    print(" Запуск. R=рестарт, Esc=выход")
    running = True
    episode = 0
    total_len = 0
    total_steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    episode += 1
                    total_len += len(env.snake)
                    total_steps += steps_taken
                    print(
                        f"#{episode} | Len: {len(env.snake)} | Steps: {steps_taken} | Avg Len: {total_len / episode:.1f}")
                    env.reset()
                    agent_state = agent.get_state(env)
                    step_budget = BASE_STEPS
                    steps_taken = 0
                    prev_len = len(env.snake)

        action = agent.get_action(agent_state)
        _, _, done, info = env.step(action)
        agent_state = agent.get_state(env)
        steps_taken += 1

        if len(env.snake) > prev_len:
            step_budget += STEP_BONUS
            prev_len = len(env.snake)

        if steps_taken >= step_budget or done:
            episode += 1
            length = info.get('length', len(env.snake))
            total_len += length
            total_steps += steps_taken
            print(f"#{episode} | Len: {length:2d} | Steps: {steps_taken:4d} | Avg Len: {total_len / episode:5.1f}")

            env.reset()
            agent_state = agent.get_state(env)
            step_budget = BASE_STEPS
            steps_taken = 0
            prev_len = len(env.snake)

    if episode > 0:
        print(
            f"\n🏁 Итого: Эпизодов {episode}, Ср. длина {total_len / episode:.1f}, Ср. шаги {total_steps / episode:.1f}")
    env.close()


if __name__ == "__main__":
    # Выбери режим:
    #train()
    play_trained_model(MODEL_PATH)  # Раскомментируй для просмотра игры