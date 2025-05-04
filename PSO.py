import numpy as np

class ParticleSwarmOptimizer:
  # Constructor:
    def __init__(self, fitness_function, bounds, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.fitness = fitness_function
        self.bounds = np.array(bounds)  # shape (n_dimensions, 2)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = self.bounds.shape[0]
        self.w, self.c1, self.c2 = w, c1, c2

        # Initializing the swarm:
        self.swarm = []
        for _ in range(num_particles):
            position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]) # Random initial position
            velocity = np.zeros(self.dim)
            particle = {
                'position': position,
                'velocity': velocity,
                'best_pos': position.copy(),
                'best_score': -np.inf
            }
            self.swarm.append(particle)

        self.global_best_pos = None
        self.global_best_score = -np.inf

  # Components:
  # fitness_function: The function to optimize (maximize).
  # bounds: List/array of shape (n_dimensions, 2), giving the lower and upper bounds for each dimension.
  # num_particles: Number of candidate solutions (particles).
  # max_iter: Number of iterations.
  # w: Inertia weight, controls how much of the previous velocity is retained.
  # c1, c2: Acceleration coefficients (cognitive and social).


  # Main function:
    def optimize(self):
        for i in range(self.max_iter):
            print(f"Iteration {i}: Best score: {self.global_best_score}")
          # Evaluating each particle for max_iter number of times
            for particle in self.swarm:
                pos = particle['position']
                score = self.fitness(pos)

                if score > particle['best_score']: # If the score is better than the particle’s previous best, we update it
                    particle['best_score'] = score
                    particle['best_pos'] = pos.copy()

                if score > self.global_best_score: # If the score is better than the global best, we update it
                    self.global_best_score = score
                    self.global_best_pos = pos.copy()

            # Updating the Velocities and Positions
            for particle in self.swarm:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim) # Generates two vectors of random values between 0 and 1 for each dimension. These are used in the velocity update rule
                cognitive = self.c1 * r1 * (particle['best_pos'] - particle['position'])
                social = self.c2 * r2 * (self.global_best_pos - particle['position'])
                particle['velocity'] = self.w * particle['velocity'] + cognitive + social
                particle['position'] += particle['velocity']
                particle['position'] = np.clip(particle['position'], self.bounds[:, 0], self.bounds[:, 1]) # Restricts each particle’s new position so that it stays within the defined bounds after velocity updates.

        return self.global_best_pos, self.global_best_score
