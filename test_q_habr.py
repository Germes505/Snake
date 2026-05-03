# test_q_habr_torch.py
"""
Тестер для QAgentHabrTorch — минимализм как в статье.
"""
import pygame
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from snake_env import SnakeEnv
from agents.q_agent_habr import QAgentHabrTorch

# ==================== НАСТРОЙКИ ====================
GRID_SIZE = 20
EPISODES = 3000
MAX_STEPS = 2000
SAVE_MODEL = True
MODEL_PATH = "q_agent_habr_torch.pt"

# Гиперпараметры КАК В СТАТЬЕ
LR = 0.01  # Как в статье
EPSILON_START = 0.8  # Как в статье (не 1.0!)
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01
GAMMA = 0.99

# Простые награды как в статье
REWARD_FOOD = 1.0
REWARD_DEATH = -1.0
REWARD_STEP = 0.01


# ==================== ОБУЧЕНИЕ ====================
def train():
    print(f"🚀 Обучение (стиль Хабра + PyTorch): {EPISODES} эпизодов")
    print(f"📊 Net: 12→1024→1024→4 | LR: {LR} | ε: {EPSILON_START}→{EPSILON_MIN}")

    start_time = time.time()
    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=20, render_mode=None)
    agent = QAgentHabrTorch(
        learning_rate=LR, epsilon=EPSILON_START, gamma=GAMMA,
        input_size=12, action_size=4
    )

    scores = []
    losses = []
    avg_window = deque(maxlen=100)
    best_score, best_ep = 0, 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        agent_state = agent.get_state(env)
        done = False

        while not done:
            action = agent.get_action(agent_state, env.direction, training=True)
            next_state_env, reward, done, info = env.step(action)
            next_state = agent.get_state(env)

            # Простые награды как в статье
            if reward == 10:
                final_reward = REWARD_FOOD
            elif reward == -10:
                final_reward = REWARD_DEATH
            else:
                final_reward = REWARD_STEP

            agent.remember(agent_state, action, final_reward, next_state, done)
            loss = agent.train_snake(batch_size=min(64, agent.memory_len))

            agent_state = next_state
            if info['steps'] >= MAX_STEPS:
                done = True

        score = info['length'] - 3
        scores.append(score)
        if loss is not None:
            losses.append(loss)
        avg_window.append(score)
        agent.update_epsilon(decay=EPSILON_DECAY, min_val=EPSILON_MIN)

        if score > best_score:
            best_score, best_ep = score, episode

        if episode % 100 == 0:
            elapsed = time.time() - start_time
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            avg = np.mean(avg_window)
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"⏱️ {int(h)}ч {int(m)}м {int(s)}с | Ep: {episode:4d} | "
                  f"Score: {score:3d} | Avg: {avg:5.2f} | ε: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | Best: {best_score} (ep {best_ep})")

    total_time = time.time() - start_time
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)

    print(f"\n🏆 Лучший результат: {best_score} (эпизод {best_ep})")
    print(f"⏱️ Общее время: {int(h)}ч {int(m)}м {int(s)}с")

    if SAVE_MODEL:
        agent.save(MODEL_PATH)

    visualize(scores, losses)
    return agent, scores


# ==================== ВИЗУАЛИЗАЦИЯ ====================
def visualize(scores, losses):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Q-Agent (Habr-style + PyTorch)', fontsize=14, fontweight='bold')

    # Score
    axs[0].plot(scores, alpha=0.2, color='gray')
    if len(scores) >= 100:
        smooth = np.convolve(scores, np.ones(100) / 100, mode='valid')
        axs[0].plot(range(99, len(scores)), smooth, color='blue', linewidth=2)
    axs[0].set_title('Score per Episode')
    axs[0].set_xlabel('Episode');
    axs[0].set_ylabel('Score')
    axs[0].grid(True, alpha=0.3)

    # Loss
    axs[1].plot(losses, alpha=0.5, color='orange')
    if len(losses) >= 100:
        smooth = np.convolve(losses, np.ones(100) / 100, mode='valid')
        axs[1].plot(range(99, len(losses)), smooth, color='red', linewidth=2)
    axs[1].set_title('Training Loss')
    axs[1].set_xlabel('Episode');
    axs[1].set_ylabel('Loss')
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('q_habr_torch_progress.png', dpi=200, bbox_inches='tight')
    print("📊 Графики: 'q_habr_torch_progress.png'")
    plt.show()


# ==================== ТЕСТ МОДЕЛИ ====================
def play_model(path=MODEL_PATH):
    agent = QAgentHabrTorch(learning_rate=LR, epsilon=0.0)
    agent.load(path)

    env = SnakeEnv(grid_size=GRID_SIZE, cell_size=30, render_mode='human')
    env.reset()
    state = agent.get_state(env)

    print("🐍 Запуск. R=рестарт, Esc=выход")

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
                    state = agent.get_state(env)

        action = agent.get_action(state, env.direction, training=False)
        state, reward, done, info = env.step(action)
        state = agent.get_state(env)

        if done:
            print(f"💀 Score: {info['length'] - 3}")
            env.reset()
            state = agent.get_state(env)

    env.close()


# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    print(f"✅ PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print("\n1 — Обучить")
    print("2 — Протестировать модель")

    choice = input("Выбор [1/2]: ").strip()
    if choice == '1':
        train()
    elif choice == '2':
        play_model()
    else:
        print("❌ Неверный выбор")