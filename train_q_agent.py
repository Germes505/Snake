# train_q_agent.py
import pygame
import pickle
import matplotlib.pyplot as plt
from collections import deque
from snake_env import SnakeEnv
from agents.q_agent import QAgent

# --- КОНФИГУРАЦИЯ ---
GRID_SIZE = 20
EPISODES = 2000  # Сколько игр сыграет агент
MAX_STEPS = 2000  # Лимит шагов за игру (чтобы не зависал)
SAVE_MODEL = True
MODEL_PATH = "q_agent_model.pkl"

# Гиперпараметры
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Скорость затухания любопытства


def train():
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=20, render_mode=None)  # render_mode=None для ускорения
    agent = QAgent(alpha=0.1, gamma=0.9, epsilon=EPSILON_START)

    score_history = []
    epsilon_history = []
    avg_score_window = deque(maxlen=50)  # Окно для скользящего среднего

    print(f"🚀 Начало обучения: {EPISODES} эпизодов...")

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        agent_state = agent.get_state(env)
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            # 1. Выбор действия
            action = agent.get_action(agent_state)

            # 2. Шаг среды
            next_state_env, reward, done, info = env.step(action)
            next_state = agent.get_state(env)
            step_count += 1

            # Штраф за время (чтобы учился есть быстрее)
            # Если еда съедена, reward уже положительный из env (+10)
            # Если просто шаг, даем маленький штраф
            if reward == 0:
                reward = -0.1
            elif done:
                reward = -10  # Смерть

            # 3. Обучение (Update Q-Table)
            agent.train(agent_state, action, reward, next_state, done)

            agent_state = next_state
            total_reward += reward

            if step_count >= MAX_STEPS:
                done = True  # Таймаут

        # Статистика эпизода
        score = info['length'] - 3  # Score = длина - 3 (начальная длина)
        avg_score_window.append(score)
        score_history.append(score)
        epsilon_history.append(agent.epsilon)

        # Затухание Epsilon
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        if episode % 50 == 0:
            avg = sum(avg_score_window) / len(avg_score_window)
            print(f"Ep: {episode} | Score: {score} | Avg: {avg:.2f} | Epsilon: {agent.epsilon:.3f}")

    # --- СОХРАНЕНИЕ И ГРАФИКИ ---
    if SAVE_MODEL:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(agent, f)
        print(f"\n💾 Модель сохранена в {MODEL_PATH}")

    # График обучения
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(score_history)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score (Length)')

    plt.subplot(1, 2, 2)
    # Скользящее среднее для сглаживания
    avg_scores = [sum(score_history[max(0, i - 100):i + 1]) / len(score_history[max(0, i - 100):i + 1])
                  for i in range(len(score_history))]
    plt.plot(avg_scores)
    plt.title('Average Score (Window=100)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.tight_layout()
    plt.show()


def play_trained_model():
    """Визуализация обученного агента"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            agent = pickle.load(f)
    except FileNotFoundError:
        print("❌ Модель не найдена! Сначала запустите train().")
        return

    # Включаем Epsilon на 0 (только умные ходы)
    agent.epsilon = 0.0

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

        # Всегда выбираем лучшее действие
        action = agent.get_action(agent_state)
        state, reward, done, info = env.step(action)
        agent_state = agent.get_state(env)

        if done:
            print(f" Game Over. Score: {info['length'] - 3}")
            env.reset()
            agent_state = agent.get_state(env)

    env.close()


if __name__ == "__main__":
    # Выбери что запустить:
    train()
    # play_trained_model() # Раскомментируй, чтобы посмотреть игру после обучения