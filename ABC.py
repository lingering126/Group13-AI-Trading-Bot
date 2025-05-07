# ABC Implementation
class ABC:
    def __init__(self, objective_func, bounds, n_bees=50, n_iterations=100, limit=50):
        """
        Artificial Bee Colony Algorithm implementation

        Parameters:
        - objective_func: Objective function to optimize
        - bounds: Parameter bounds [(min, max), ...]
        - n_bees: Number of bees
        - n_iterations: Number of iterations
        - limit: Maximum number of trials
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_bees = n_bees
        self.n_iterations = n_iterations
        self.limit = limit
        self.dim = len(bounds)

        # Initialize bee positions
        self.positions = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (n_bees, self.dim)
        )

        # Initialize fitness
        self.fitness = np.zeros(n_bees)
        self.trials = np.zeros(n_bees)
        self.best_solution = None
        self.best_fitness = float('-inf')

    def evaluate_fitness(self, position):
        """Evaluate fitness of a position"""
        return self.objective_func(position)

    def employed_bee_phase(self):
        """Employed bee phase"""
        for i in range(self.n_bees):
            # Randomly select a dimension
            j = np.random.randint(self.dim)
            # Randomly select a different bee
            k = np.random.randint(self.n_bees)
            while k == i:
                k = np.random.randint(self.n_bees)

            # Generate new solution
            new_position = self.positions[i].copy()
            phi = np.random.uniform(-1, 1)
            new_position[j] = self.positions[i][j] + phi * (self.positions[i][j] - self.positions[k][j])

            # Ensure new solution is within bounds
            new_position[j] = np.clip(new_position[j], self.bounds[j][0], self.bounds[j][1])

            # Evaluate new solution
            new_fitness = self.evaluate_fitness(new_position)

            # Update if new solution is better
            if new_fitness > self.fitness[i]:
                self.positions[i] = new_position
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bee_phase(self):
        """Onlooker bee phase"""
        # Calculate selection probabilities
        fitness_sum = np.sum(self.fitness)
        if fitness_sum == 0:
            probabilities = np.ones(self.n_bees) / self.n_bees
        else:
            probabilities = self.fitness / fitness_sum

        for _ in range(self.n_bees):
            # Roulette wheel selection
            i = np.random.choice(self.n_bees, p=probabilities)

            # Randomly select a dimension
            j = np.random.randint(self.dim)
            # Randomly select a different bee
            k = np.random.randint(self.n_bees)
            while k == i:
                k = np.random.randint(self.n_bees)

            # Generate new solution
            new_position = self.positions[i].copy()
            phi = np.random.uniform(-1, 1)
            new_position[j] = self.positions[i][j] + phi * (self.positions[i][j] - self.positions[k][j])

            # Ensure new solution is within bounds
            new_position[j] = np.clip(new_position[j], self.bounds[j][0], self.bounds[j][1])

            # Evaluate new solution
            new_fitness = self.evaluate_fitness(new_position)

            # Update if new solution is better
            if new_fitness > self.fitness[i]:
                self.positions[i] = new_position
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def scout_bee_phase(self):
        """Scout bee phase"""
        for i in range(self.n_bees):
            if self.trials[i] >= self.limit:
                # Reinitialize the position of this bee
                self.positions[i] = np.random.uniform(
                    [b[0] for b in self.bounds],
                    [b[1] for b in self.bounds],
                    self.dim
                )
                self.fitness[i] = self.evaluate_fitness(self.positions[i])
                self.trials[i] = 0

    def optimize(self):
        """Execute optimization"""
        # Initialize fitness
        for i in range(self.n_bees):
            self.fitness[i] = self.evaluate_fitness(self.positions[i])

        # Record best solution
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.positions[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        # Iterative optimization
        for iteration in range(self.n_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            # Update best solution
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_solution = self.positions[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness
