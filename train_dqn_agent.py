# train_dqn_agent.py
import pygame
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent

# --- КОНФИГУРАЦИЯ ---
GRID_SIZE = 20
EPISODES = 1000  # Количество игр для обучения
MAX_STEPS = 5000  # Лимит шагов за эпизод
SAVE_MODEL = True
MODEL_PATH = "dqn_agent_model.pt"

# Гиперпараметры
LR = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999  # Медленное затухание
BUFFER_SIZE = 100000
BATCH_SIZE = 64
TARGET_UPDATE = 1000


def smooth_curve(data, window_size=100):
    """Сглаживание кривой скользящим средним."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def train():
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=20, render_mode=None)
    agent = DQNAgent(
        state_size=GRID_SIZE * GRID_SIZE,
        action_size=4,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update=TARGET_UPDATE
    )

    score_history = []
    loss_history = []
    epsilon_history = []
    avg_score_window = deque(maxlen=100)

    best_score = 0
    best_episode = 0

    print(f"🚀 Начало обучения DQN: {EPISODES} эпизодов...")
    print(f"📊 Гиперпараметры: lr={LR}, γ={GAMMA}, ε_decay={EPSILON_DECAY}")

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        agent_state = agent.get_state(env)
        total_reward = 0
        done = False
        step_count = 0
        episode_loss = 0

        while not done:
            # Выбор действия
            action = agent.get_action(agent_state, training=True)

            # Шаг среды
            next_state_env, reward, done, info = env.step(action)
            next_state = agent.get_state(env)
            step_count += 1

            # Функция наград
            if reward == 0:
                reward = 0.1  # Поощрение выживания
            elif reward > 0:
                reward = 20  # Большая награда за еду
            elif done:
                reward = -20  # Штраф за смерть

            # Сохранение опыта
            agent.remember(agent_state, action, reward, next_state, done)

            # Обучение
            if len(agent.memory) >= BATCH_SIZE:
                loss = agent.learn()
                if loss:
                    episode_loss += loss

            agent_state = next_state
            total_reward += reward

            if step_count >= MAX_STEPS:
                done = True

        # Статистика
        score = info['length'] - 3
        score_history.append(score)
        epsilon_history.append(agent.epsilon)
        avg_score_window.append(score)
        loss_history.append(episode_loss / max(step_count, 1))

        # Обновление epsilon
        agent.update_epsilon()

        # Отслеживание рекорда
        if score > best_score:
            best_score = score
            best_episode = episode
            if SAVE_MODEL:
                agent.save(MODEL_PATH.replace('.pt', '_best.pt'))

        # Вывод прогресса
        if episode % 100 == 0:
            avg = sum(avg_score_window) / len(avg_score_window)
            avg_loss = np.mean(loss_history[-100:])
            print(f"Ep: {episode:5d} | Score: {score:3d} | Avg: {avg:6.2f} | "
                  f"ε: {agent.epsilon:.4f} | Loss: {avg_loss:.4f} | Best: {best_score}")

    print(f"\n🏆 Лучший результат: {best_score} (эпизод {best_episode})")

    # Сохранение финальной модели
    if SAVE_MODEL:
        agent.save(MODEL_PATH)

    # Визуализация
    visualize_training(score_history, loss_history, epsilon_history)


def visualize_training(scores, losses, epsilons):
    """Продвинутая визуализация результатов обучения (исправлены размеры массивов)."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Agent Training Progress', fontsize=16, fontweight='bold')

    # 1. Score per Episode
    axs[0, 0].plot(scores, alpha=0.2, label='Raw', color='gray')
    smooth_scores = smooth_curve(scores, 100)
    # ✅ ИСПРАВЛЕНО: range заканчивается на len(scores), а не len(scores)+1
    axs[0, 0].plot(range(99, len(scores)), smooth_scores,
                   label='Moving Avg (100)', color='blue', linewidth=2)
    axs[0, 0].set_title('Score per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Loss per Episode
    axs[0, 1].plot(losses, alpha=0.5, color='orange')
    smooth_loss = smooth_curve(losses, 100)
    # ✅ ИСПРАВЛЕНО: тот же фикс
    axs[0, 1].plot(range(99, len(losses)), smooth_loss,
                   color='red', linewidth=2)
    axs[0, 1].set_title('Training Loss')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Loss (MSE)')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Epsilon Decay
    axs[1, 0].plot(epsilons, color='green', linewidth=2)
    axs[1, 0].set_title('Exploration Rate (Epsilon)')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Epsilon')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Distribution of Scores
    axs[1, 1].hist(scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axs[1, 1].axvline(np.mean(scores), color='red', linestyle='dashed',
                      linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    axs[1, 1].set_title('Distribution of Scores')
    axs[1, 1].set_xlabel('Score')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_training_progress.png', dpi=300, bbox_inches='tight')
    print("📊 Графики сохранены в 'dqn_training_progress.png'")
    plt.show()


def play_trained_model(model_path=MODEL_PATH):
    """Визуализация обученного агента."""
    import torch
    agent = DQNAgent(state_size=GRID_SIZE * GRID_SIZE, action_size=4)
    agent.load(model_path)
    agent.epsilon = 0.0  # Только детерминированные действия

    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=30, render_mode='human')
    state = env.reset()
    agent_state = agent.get_state(env)

    print("🐍 Запуск DQN-агента. Нажмите R для рестарта.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    agent_state = agent.get_state(env)

        action = agent.get_action(agent_state, training=False)
        state, reward, done, info = env.step(action)
        agent_state = agent.get_state(env)

        if done:
            print(f"💀 Game Over. Score: {info['length']}, {env.steps}")
            env.reset()
            agent_state = agent.get_state(env)

    env.close()


if __name__ == "__main__":
    # Выбери режим:
    # train()
    play_trained_model()  # Раскомментируй для просмотра игры