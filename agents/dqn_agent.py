# agents/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQN(nn.Module):
    """
    Нейросетевая архитектура для аппроксимации Q-функции.
    Вход: вектор состояния (400 элементов для поля 20×20)
    Выход: 4 Q-значения (UP, DOWN, LEFT, RIGHT)
    """

    def __init__(self, input_size=400, output_size=4, hidden_sizes=[256, 128, 64]):
        super(DQN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Инициализация весов (Xavier)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Буфер воспроизведения опыта (Experience Replay).
    Хранит переходы (s, a, r, s', done) и отдаёт случайные мини-батчи.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Сохраняет переход в буфер."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Возвращает случайный мини-батч."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Конвертируем в тензоры PyTorch
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Агент, обучающийся методом Deep Q-Learning.
    """

    def __init__(
            self,
            state_size=400,
            action_size=4,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.9999,
            buffer_size=100000,
            batch_size=64,
            target_update=1000,
            device=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Устройство: CUDA если есть, иначе CPU
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 DQN Agent initialized on device: {self.device}")

        # Основная и целевая сети
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Целевая сеть только для forward

        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Буфер памяти
        self.memory = ReplayBuffer(buffer_size)

        self.steps = 0  # Счётчик шагов для обновления целевой сети

    def get_state(self, env):
        """
        Преобразует состояние среды в тензор для нейросети.
        Возвращает бинарную матрицу 20×20, линеаризованную в вектор 400.
        """
        grid = np.zeros((env.grid_size, env.grid_size), dtype=np.float32)

        # Помечаем тело змейки
        for segment in env.snake:
            grid[segment[0], segment[1]] = 0.5

        # Голова — отдельная метка
        head = env.snake[0]
        grid[head[0], head[1]] = 1.0

        # Еда
        if env.food:
            grid[env.food[0], env.food[1]] = 0.8

        # Линеаризуем в вектор
        return grid.flatten()

    def get_action(self, state, training=True):
        """
        Возвращает действие по ε-жадной стратегии.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Сохраняет опыт в буфер."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Один шаг обучения: выбор мини-батча, вычисление потери, обновление весов.
        """
        if len(self.memory) < self.batch_size:
            return  # Недостаточно данных для обучения

        # Выбор мини-батча
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a) для текущих состояний
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Целевые значения: r + γ * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Потеря и оптимизация
        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Обновление целевой сети
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def update_epsilon(self):
        """Экспоненциальное затухание ε."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Сохраняет модель на диск."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"💾 Модель сохранена в {path}")

    def load(self, path):
        """Загружает модель с диска."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['q_network'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)
        print(f"📥 Модель загружена из {path}")