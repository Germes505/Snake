# agents/ga_agent.py
"""
Генетический алгоритм для змейки (Neuroevolution).
Использует чистый NumPy для максимальной скорости оценки популяции.
Архитектура сети: 12 → 24 → 4
"""
import numpy as np
import random
import pickle


class GANet:
    """Легковесная нейросеть на NumPy для быстрой оценки"""

    def __init__(self, input_size=12, hidden_size=24, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._init_shapes()

    def _init_shapes(self):
        self.W1_size = self.input_size * self.hidden_size
        self.b1_size = self.hidden_size
        self.W2_size = self.hidden_size * self.output_size
        self.b2_size = self.output_size
        self.total_params = self.W1_size + self.b1_size + self.W2_size + self.b2_size

    def set_weights(self, genome):
        """Распаковка плоского вектора весов в матрицы"""
        self.W1 = genome[:self.W1_size].reshape(self.input_size, self.hidden_size)
        self.b1 = genome[self.W1_size:self.W1_size + self.b1_size]
        self.W2 = genome[self.W1_size + self.b1_size:self.W1_size + self.b1_size + self.W2_size].reshape(
            self.hidden_size, self.output_size)
        self.b2 = genome[-self.b2_size:]

    def predict(self, state):
        """Прямой проход (Forward Pass)"""
        if state.ndim == 1: state = state.reshape(1, -1)
        h = state @ self.W1 + self.b1
        h = np.maximum(0, h)  # ReLU
        return h @ self.W2 + self.b2

    def get_action(self, state, env_direction):
        """Выбор действия с маской запрещённого разворота"""
        q_values = self.predict(state)[0]
        opp = env_direction ^ 1  # XOR для нахождения противоположного направления
        q_values[opp] = -np.inf  # Жёсткая маска
        return np.argmax(q_values)


class GAAgent:
    """Управление популяцией и эволюцией"""

    def __init__(self, pop_size=100, elite_count=5, mutation_rate=0.1,
                 mutation_sigma=0.1, crossover_prob=0.5, tournament_k=3,
                 input_size=12, hidden_size=24, output_size=4):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_prob = crossover_prob
        self.tournament_k = tournament_k

        self.net = GANet(input_size, hidden_size, output_size)
        self.population = []
        self.fitness_history = []
        self.best_genome = None
        self.best_fitness = -np.inf
        self.generation = 0

    def initialize_population(self):
        """Генерация случайной популяции"""
        self.population = [np.random.uniform(-0.5, 0.5, self.net.total_params) for _ in range(self.pop_size)]

    def _get_state(self, env):
        """Извлечение 12 признаков (совместимо с DQN)"""
        head = env.snake[0]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cur = dirs[env.direction]

        # ✅ ИСПРАВЛЕННЫЕ формулы поворота
        pt_s = (head[0] + cur[0], head[1] + cur[1])
        pt_r = (head[0] + cur[1], head[1] - cur[0])
        pt_l = (head[0] - cur[1], head[1] + cur[0])

        danger = [1.0 if self._is_coll(pt, env) else 0.0 for pt in [pt_s, pt_r, pt_l]]

        food = env.food if env.food else (0, 0)
        food_dir = [0.0] * 4
        if food[0] < head[0]:
            food_dir[0] = 1.0
        elif food[0] > head[0]:
            food_dir[1] = 1.0
        elif food[1] < head[1]:
            food_dir[2] = 1.0
        else:
            food_dir[3] = 1.0

        wall_dist = [self._dist_to_wall(head, d, env) / env.grid_size for d in dirs]
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

    def evaluate_individual(self, genome, env, max_steps=2000):
        """Оценка одной особи: одна игра = фитнес"""
        self.net.set_weights(genome)
        env.reset()
        agent_state = self._get_state(env)
        done = False
        steps = 0

        while not done:
            action = self.net.get_action(agent_state, env.direction)
            _, _, done, info = env.step(action)
            agent_state = self._get_state(env)
            steps += 1
            if steps >= max_steps:
                done = True

        return info['length'] - 3

    def select_tournament(self, fitness):
        """Турнирная селекция (k=3)"""
        indices = random.sample(range(self.pop_size), self.tournament_k)
        best_idx = max(indices, key=lambda i: fitness[i])
        return self.population[best_idx].copy()

    def crossover(self, p1, p2):
        """Равномерный кроссовер"""
        if random.random() > self.crossover_prob:
            return p1.copy()
        mask = np.random.choice([True, False], size=len(p1))
        return np.where(mask, p1, p2)

    def mutate(self, genome):
        """Гауссова мутация"""
        mutation_mask = np.random.random(genome.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_sigma, genome.shape)
        genome[mutation_mask] += noise[mutation_mask]
        return genome

    def evolve(self, fitness):
        """Один шаг эволюции"""
        sorted_indices = np.argsort(fitness)[::-1]
        sorted_pop = [self.population[i] for i in sorted_indices]

        # Элитизм
        new_population = sorted_pop[:self.elite_count]

        # Запоминаем абсолютного рекордсмена
        if fitness[sorted_indices[0]] > self.best_fitness:
            self.best_fitness = fitness[sorted_indices[0]]
            self.best_genome = sorted_pop[0].copy()

        # Создание потомства
        while len(new_population) < self.pop_size:
            p1 = self.select_tournament(fitness)
            p2 = self.select_tournament(fitness)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        self.fitness_history.append({
            'best': self.best_fitness,
            'avg': np.mean(fitness),
            'gen_max': fitness[sorted_indices[0]]
        })

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'best_genome': self.best_genome,
                'fitness_history': self.fitness_history,
                'generation': self.generation
            }, f)
        print(f"💾 GA модель сохранена в {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.best_genome = data['best_genome']
        self.fitness_history = data['fitness_history']
        self.generation = data['generation']
        self.best_fitness = data['fitness_history'][-1]['best'] if data['fitness_history'] else -np.inf
        print(f"📥 GA модель загружена из {path}")