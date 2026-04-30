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
EPISODES = 100000  # Увеличили количество игр
MAX_STEPS = 40000

# 🎯 УЛУЧШЕННЫЕ ГИПЕРПАРАМЕТРЫ
ALPHA = 0.001  # Чуть выше скорость обучения
GAMMA = 0.99  # Больше внимания будущему
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999  # МЕДЛЕННЕЕ затухание (агент дольше исследует)

SAVE_MODEL = True
MODEL_PATH = "q_agent_model_improved.pkl"

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
    avg_score_window = deque(maxlen=100)

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
        step_count = 0

        while not done:
            action = agent.get_action(agent_state)
            next_state_env, reward, done, info = env.step(action)
            next_state = agent.get_state(env)
            step_count += 1

            # 🎯 УЛУЧШЕННАЯ ФУНКЦИЯ НАГРАД
            if reward == 0:
                reward = 0.001  # Меньше штраф за время (чтобы не боялся жить долго)
            elif reward > 0:
                reward = 20  # Больше награда за еду
            elif done:
                reward = -150  # Больше штраф за смерть

            agent.train(agent_state, action, reward, next_state, done)

            agent_state = next_state
            total_reward += reward

            if step_count >= MAX_STEPS:
                done = True

        # Статистика
        score = info['length'] - 3
        score_history.append(score)
        epsilon_history.append(agent.epsilon)
        avg_score_window.append(score)

        # Отслеживание рекорда
        if score > best_score:
            best_score = score
            best_episode = episode

        # Затухание Epsilon
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        # Вывод прогресса
        if episode % 100 == 0:
            avg = sum(avg_score_window) / len(avg_score_window)
            print(f"Ep: {episode:4d} | Score: {score:3d} | Avg: {avg:6.2f} | "
                  f"ε: {agent.epsilon:.4f} | Best: {best_score} (ep {best_episode})")

        if score > best_score:
            best_score = score
            best_episode = episode
            # СОХРАНЯЕМ ЛУЧШУЮ МОДЕЛЬ
            with open("q_agent_best.pkl", "wb") as f:
                pickle.dump(agent, f)

    print(f"\n🏆 Лучший результат: {best_score} (эпизод {best_episode})")

    # --- СОХРАНЕНИЕ ---
    if SAVE_MODEL:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(agent, f)
        print(f"💾 Модель сохранена в {MODEL_PATH}")

    # --- 📊 УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    visualize_training(score_history, epsilon_history)


def visualize_training(scores, epsilons):
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

    # 3. Распределение результатов
    axs[1, 0].hist(scores, bins=50, color='green', alpha=0.7, edgecolor='black')
    axs[1, 0].axvline(np.mean(scores), color='red', linestyle='dashed',
                      linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    axs[1, 0].set_title('Distribution of Scores')
    axs[1, 0].set_xlabel('Score')
    axs[1, 0].set_ylabel('Frequency')
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
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("📊 Графики сохранены в 'training_progress.png'")
    plt.show()


def play_trained_model(model_path="q_agent_model_improved.pkl"):
    """Визуализация обученного агента"""
    try:
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Модель '{model_path}' не найдена!")
        return

    agent.epsilon = 0.0  # Только умные ходы

    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=30, render_mode='human')
    state = env.reset()
    agent_state = agent.get_state(env)

    print("🐍 Запуск обученного агента. Нажмите R для рестарта.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    agent_state = agent.get_state(env)

        action = agent.get_action(agent_state)
        state, reward, done, info = env.step(action)
        agent_state = agent.get_state(env)

        if done:
            print(f"💀 Game Over. Score: {info['length'] - 3}")
            env.reset()
            agent_state = agent.get_state(env)

    env.close()


if __name__ == "__main__":
    # Выбери режим:
    # train()
    play_trained_model()  # Раскомментируй для просмотра игры