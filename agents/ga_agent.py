# agents/ga_agent.py
"""
Island Model GA + 48 нейронов + Tanh + Адаптивная мутация
"""
import numpy as np
import random
import pickle
from snake_env import SnakeEnv  # Нужно для миграции


class GANet:
    def __init__(self, input_size=12, hidden_size=48, output_size=4, base_steps=400, step_bonus=400):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.base_steps = base_steps
        self.step_bonus = step_bonus
        self._init_shapes()

    def _init_shapes(self):
        self.W1_size = self.input_size * self.hidden_size
        self.b1_size = self.hidden_size
        self.W2_size = self.hidden_size * self.output_size
        self.b2_size = self.output_size
        self.total_params = self.W1_size + self.b1_size + self.W2_size + self.b2_size

    def set_weights(self, genome):
        self.W1 = genome[:self.W1_size].reshape(self.input_size, self.hidden_size)
        self.b1 = genome[self.W1_size:self.W1_size + self.b1_size]
        self.W2 = genome[self.W1_size + self.b1_size:self.W1_size + self.b1_size + self.W2_size].reshape(
            self.hidden_size, self.output_size)
        self.b2 = genome[-self.b2_size:]

    def predict(self, state):
        if state.ndim == 1: state = state.reshape(1, -1)
        h = state @ self.W1 + self.b1
        h = np.tanh(h)  # Tanh стабильнее для эволюции
        return h @ self.W2 + self.b2

    def get_action(self, state, env_direction):
        q_values = self.predict(state)[0]
        opp = env_direction ^ 1
        q_values[opp] = -np.inf
        return np.argmax(q_values)


class GAAgent:
    def __init__(self, pop_size=120, elite_count=3, mutation_rate=0.1,
                 mutation_sigma=0.15, crossover_prob=0.7, tournament_k=3,
                 input_size=12, hidden_size=48, output_size=4,
                 n_islands=3, migration_interval=20, migration_size=2):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_prob = crossover_prob
        self.tournament_k = tournament_k
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size

        self.net = GANet(input_size, hidden_size, output_size)
        self.islands = [[] for _ in range(n_islands)]
        self.fitness_history = []
        self.best_genome = None
        self.best_fitness = -999.0
        self.generation = 0
        self.stagnation_counter = 0

    def initialize_population(self):
        per_island = self.pop_size // self.n_islands
        for i in range(self.n_islands):
            self.islands[i] = [np.random.uniform(-0.5, 0.5, self.net.total_params)
                               for _ in range(per_island)]

    def _get_state(self, env):
        head = env.snake[0]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cur = dirs[env.direction]
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
            r += dr;
            c += dc
        return steps

    def evaluate_individual(self, genome, env, base_steps=400, step_bonus=150):
        """
        Оценка особи с динамическим лимитом шагов.

        Параметры:
        ----------
        genome : np.ndarray
            Вектор весов нейронной сети.
        env : SnakeEnv
            Экземпляр игровой среды.
        base_steps : int, optional (default=400)
            Базовый бюджет шагов на эпизод.
        step_bonus : int, optional (default=400)
            Дополнительные шаги за каждое съеденное яблоко.

        Возвращает:
        -----------
        float
            Значение фитнес-функции.
        """
        self.net.set_weights(genome)
        env.reset()

        # Динамический лимит шагов
        step_budget = base_steps  # Текущий бюджет
        steps_taken = 0  # Счётчик сделанных шагов
        food_eaten = 0  # Счётчик съеденной еды

        agent_state = self._get_state(env)
        done = False

        while not done:
            # Проверка: исчерпан ли бюджет шагов
            if steps_taken >= step_budget:
                done = True
                break

            action = self.net.get_action(agent_state, env.direction)

            # Запоминаем позицию головы до шага для детекции поедания
            old_head = env.snake[0].copy() if hasattr(env.snake[0], 'copy') else tuple(env.snake[0])

            _, reward, done, info = env.step(action)
            agent_state = self._get_state(env)
            steps_taken += 1

            # Детекция поедания: если голова изменилась и награда положительная
            if reward > 0 or (env.snake[0] != old_head and len(env.snake) > info.get('length', len(env.snake))):
                food_eaten += 1
                step_budget = step_bonus  # 🔥 Увеличиваем бюджет на step_bonus шагов

        # Фитнес-функция: базовый счёт + бонус за выживание
        base_score = len(env.snake)
        survival_bonus = min(steps_taken * 0.002, 3.0)

        return float(base_score + survival_bonus), steps_taken
    def select_tournament(self, population, fitness):
        indices = random.sample(range(len(population)), self.tournament_k)
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx].copy()

    def crossover(self, p1, p2):
        if random.random() > self.crossover_prob: return p1.copy()
        mask = np.random.choice([True, False], size=len(p1))
        return np.where(mask, p1, p2)

    def mutate(self, genome, sigma=None, rate=None):
        sigma = sigma or self.mutation_sigma
        rate = rate or self.mutation_rate
        mask = np.random.random(genome.shape) < rate
        genome[mask] += np.random.normal(0, sigma, genome.shape)[mask]
        return genome

    def evolve_island(self, island, fitness):
        sorted_idx = np.argsort(fitness)[::-1]
        sorted_pop = [island[i] for i in sorted_idx]
        new_pop = sorted_pop[:self.elite_count]

        while len(new_pop) < len(island):
            p1 = self.select_tournament(island, fitness)
            p2 = self.select_tournament(island, fitness)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)
        return new_pop

    def migrate(self):
        # Простая миграция: обмен лучшими генами между соседними островами
        for i in range(self.n_islands):
            src = self.islands[i]
            dst = self.islands[(i + 1) % self.n_islands]
            # Берем 2 лучших из src и заменяем 2 худших в dst
            # (Предполагаем, что в начале списка лежат лучшие после сортировки в evolve_island)
            for k in range(self.migration_size):
                dst[-(k + 1)] = src[k].copy()

    def evolve(self, fitness_by_island):
        all_fitness = []
        for i, island in enumerate(self.islands):
            # Применение эволюционных операторов (селекция, кроссовер, мутация) к i-му острову
            new_island = self.evolve_island(island, fitness_by_island[i])
            self.islands[i] = new_island
            # Накопление фитнес-значений для расчёта глобальной статистики
            all_fitness.extend(fitness_by_island[i])

        # Периодическая миграция особей между островами для обмена генетическим материалом
        # и предотвращения преждевременной сходимости отдельных субпопуляций
        if self.generation > 0 and self.generation % self.migration_interval == 0:
            self.migrate()

        # Определение максимального значения приспособленности в текущем поколении
        current_best = max(all_fitness) if all_fitness else 0
        # Обновление глобального рекорда и сохранение генома лучшей особи
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            for i, island in enumerate(self.islands):
                if max(fitness_by_island[i]) == current_best:
                    best_idx = np.argmax(fitness_by_island[i])
                    self.best_genome = island[best_idx].copy()
                    break
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Адаптивная регуляция интенсивности поиска: при длительном застое (>50 поколений)
        # постепенно увеличивается стандартное отклонение мутации для выхода из локального оптимума
        if self.stagnation_counter > 50:
            self.mutation_sigma = min(0.4, self.mutation_sigma * 1.2)

        self.generation += 1
        # Сохранение метрик поколения для последующего анализа динамики сходимости
        self.fitness_history.append({
            'best': float(self.best_fitness),
            'avg': float(np.mean(all_fitness)),
            'gen_max': current_best
        })

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'best_genome': self.best_genome,
                'fitness_history': self.fitness_history,
                'generation': self.generation,
                'config': {'hidden_size': self.net.hidden_size, 'n_islands': self.n_islands}
            }, f)
        print(f"💾 Saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.best_genome = data['best_genome']
        self.fitness_history = data['fitness_history']
        self.generation = data['generation']
        self.best_fitness = data['fitness_history'][-1]['best'] if data['fitness_history'] else -999.0
        if 'config' in data:
            self.net = GANet(hidden_size=data['config'].get('hidden_size', 48))
        print(f"📥 Loaded from {path}")