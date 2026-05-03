# agents/q_agent_habr_torch.py
"""
Адаптация кода из статьи Хабра (№789218) под PyTorch.
Архитектура: 12 → 1024 → 1024 → 4, как в оригинале.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init


class QNetHabr(nn.Module):
    """Сеть ТОЧНО как в статье: широкая, с инициализацией uniform(-1,1)"""

    def __init__(self, input_size=12, output_size=4):
        super(QNetHabr, self).__init__()

        # Слои как в статье
        self.inp = nn.Linear(input_size, 1024)
        self.hidden1 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, output_size)

        # Инициализация как в статье: uniform(-1, 1)
        init.uniform_(self.inp.weight, -1, 1)
        init.uniform_(self.inp.bias, -1, 1)
        init.uniform_(self.hidden1.weight, -1, 1)
        init.uniform_(self.hidden1.bias, -1, 1)
        init.uniform_(self.out.weight, -1, 1)
        init.uniform_(self.out.bias, -1, 1)

        self.activation = nn.LeakyReLU()  # Как в статье

    def forward(self, x):
        x = self.activation(self.inp(x))
        x = self.activation(self.hidden1(x))
        return self.out(x)


class QAgentHabrTorch:
    """Агент с обучением как в статье — минимализм"""

    def __init__(self, learning_rate=0.01, epsilon=0.3, gamma=0.99,
                 device=None, input_size=12, action_size=4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 QAgentHabrTorch on {self.device} | Net: 12→1024→1024→{action_size}")

        self.network = QNetHabr(input_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Простая память как в статье
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_next_states = []
        self.memory_isdones = []
        self.memory_len = 0

        self.loses = []
        self.rewards_per_epoch = []

    def get_state(self, env):
        """12 признаков — совместимо со статьёй и твоим env"""
        head = env.snake[0]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cur = dirs[env.direction]

        # Опасность в 3 направлениях относительно взгляда
        pt_s = (head[0] + cur[0], head[1] + cur[1])
        pt_l = (head[0] - cur[1], head[1] + cur[0])
        pt_r = (head[0] + cur[1], head[1] - cur[0])
        danger = [
            1.0 if self._is_coll(pt_s, env) else 0.0,
            1.0 if self._is_coll(pt_r, env) else 0.0,
            1.0 if self._is_coll(pt_l, env) else 0.0
        ]

        # Направление к еде (4 бита: UP, DOWN, LEFT, RIGHT)
        food = env.food
        food_dir = [0.0, 0.0, 0.0, 0.0]
        if food[0] < head[0]:
            food_dir[0] = 1.0
        elif food[0] > head[0]:
            food_dir[1] = 1.0
        elif food[1] < head[1]:
            food_dir[2] = 1.0
        else:
            food_dir[3] = 1.0

        # Дистанция до стен (нормализованная 0..1)
        wall_dist = [self._dist_to_wall(head, d, env) / env.grid_size for d in dirs]

        # Нормализованная длина змейки
        length_norm = len(env.snake) / (env.grid_size * env.grid_size)

        return np.array(danger + food_dir + wall_dist + [length_norm], dtype=np.float32)

    def _is_coll(self, pt, env):
        r, c = pt
        if not (0 <= r < env.grid_size and 0 <= c < env.grid_size): return True
        return pt in env.snake

    def _dist_to_wall(self, head, direction, env):
        r, c = head
        dr, dc = direction
        steps = 0
        while 0 <= r + dr < env.grid_size and 0 <= c + dc < env.grid_size:
            steps += 1
            r += dr
            c += dc
        return steps

    def get_action(self, state, env_direction, training=True):
        """ε-greedy с маской разворота на 180°"""
        opp = env_direction ^ 1
        if training and random.random() < self.epsilon:
            valid = [a for a in range(self.action_size) if a != opp]
            return random.choice(valid)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.network(s).cpu().numpy()[0]
            q[opp] = -np.inf  # Запрещаем разворот
            return np.argmax(q)

    def remember(self, state, action, reward, next_state, done):
        """Простое сохранение как в статье"""
        self.memory_states.append(state)
        self.memory_actions.append(action)
        self.memory_rewards.append(reward)
        self.memory_next_states.append(next_state)
        self.memory_isdones.append(0.0 if done else 1.0)
        self.memory_len = len(self.memory_states)

    def samplebatch(self, batch_size=None):
        """Простая выборка последних элементов (FIFO)"""
        if batch_size is None or batch_size >= self.memory_len:
            idx = slice(None)
        else:
            idx = slice(-batch_size, None)

        states = np.array(self.memory_states[idx])
        actions = np.array(self.memory_actions[idx])
        rewards = np.array(self.memory_rewards[idx])
        next_states = np.array(self.memory_next_states[idx])
        isdones = np.array(self.memory_isdones[idx])

        return (torch.FloatTensor(states).to(self.device),
                torch.LongTensor(actions).to(self.device),
                torch.FloatTensor(rewards).to(self.device),
                torch.FloatTensor(next_states).to(self.device),
                torch.FloatTensor(isdones).to(self.device))

    def train_snake(self, batch_size=None):
        """Обучение ТОЧНО как в статье — минималистичное"""
        if self.memory_len == 0:
            return None

        states, actions, rewards, next_states, isdones = self.samplebatch(batch_size)
        memory_len = states.shape[0]

        # Прямой проход
        neuro_now = self.network(states)
        neuro_next = self.network(next_states)

        # Q-значения для выбранных действий
        q_now = neuro_now[range(memory_len), actions]

        # Целевые значения: r + γ * max(Q(s')) * is_done
        q_next_max = torch.max(neuro_next, dim=1)[0]
        target = rewards + self.gamma * q_next_max * isdones

        # Loss и оптимизация
        loss = self.criterion(target, q_now)
        self.loses.append(loss.cpu().item())
        self.rewards_per_epoch.append(torch.sum(rewards.cpu()).item())

        self.optimizer.zero_grad()
        loss.backward()

        # Хак из статьи: если градиент слишком маленький — добавляем шум
        if self.network.inp.weight.grad is not None:
            if self.network.inp.weight.grad.norm() < 0.0001:
                noise = torch.randn_like(self.network.inp.weight.grad) * 0.001
                self.network.inp.weight.grad.data += noise.to(self.device)

        self.optimizer.step()
        return loss.cpu().item()

    def update_epsilon(self, decay=0.9995, min_val=0.01):
        if self.epsilon > min_val:
            self.epsilon *= decay

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'epsilon': self.epsilon,
            'loses': self.loses[-1000:],
            'rewards': self.rewards_per_epoch[-1000:]
        }, path)
        print(f"💾 Saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt['network'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.loses = ckpt.get('loses', [])
        self.rewards_per_epoch = ckpt.get('rewards', [])
        print(f"📥 Loaded from {path}")