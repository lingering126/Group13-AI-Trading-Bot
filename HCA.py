# HCA Implementation
class HCA:
    def __init__(self, objective_func, bounds, step_size=0.1, n_iterations=1000):
        """
        Hill Climbing Algorithm implementation

        Parameters:
        - objective_func: Objective function to optimize
        - bounds: Parameter bounds [(min, max), ...]
        - step_size: Size of step for parameter updates
        - n_iterations: Number of iterations
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.step_size = step_size
        self.n_iterations = n_iterations
        self.dim = len(bounds)

        # Initialize current position
        self.current_position = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        )

        # Initialize current fitness
        self.current_fitness = self.objective_func(self.current_position)

    def generate_neighbor(self, position):
        """Generate a neighboring solution"""
        neighbor = position.copy()

        # Randomly select a dimension to modify
        dim = np.random.randint(self.dim)

        # Generate a random step
        step = np.random.uniform(-self.step_size, self.step_size)

        # Update the selected dimension
        neighbor[dim] += step

        # Ensure the new value is within bounds
        neighbor[dim] = np.clip(neighbor[dim], self.bounds[dim][0], self.bounds[dim][1])

        return neighbor

    def optimize(self):
        """Execute optimization"""
        best_position = self.current_position.copy()
        best_fitness = self.current_fitness

        for iteration in range(self.n_iterations):
            # Generate a neighbor
            neighbor = self.generate_neighbor(self.current_position)

            # Evaluate the neighbor
            neighbor_fitness = self.objective_func(neighbor)

            # If the neighbor is better, move to it
            if neighbor_fitness > self.current_fitness:
                self.current_position = neighbor
                self.current_fitness = neighbor_fitness

                # Update best solution if necessary
                if neighbor_fitness > best_fitness:
                    best_position = neighbor.copy()
                    best_fitness = neighbor_fitness

            print(
                f"Iteration {iteration + 1}/{self.n_iterations}, Current fitness: {self.current_fitness:.2f}, Best fitness: {best_fitness:.2f}")

        return best_position, best_fitness
