# PSO Invoke

data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

prices = training_data[1]

#EUSHA
# Defining the bounds for the parameters (14 total)
bounds = [
    [0, 1],    # short_wma_weight_1
    [1, 500], # short_wma_length_1
    [0, 1],    # short_wma_weight_2
    [1, 500], # short_wma_length_2
    [0, 1],    # short_wma_weight_3
    [1, 500], # short_wma_length_3
    [0.01, 1],    # short_wma_alpha for EMA
    [0, 1],    # long_wma_weight_1
    [1, 500], # long_wma_length_1
    [0, 1],    # long_wma_weight_2
    [1, 500], # long_wma_length_2
    [0, 1],    # long_wma_weight_3
    [1, 500], # long_wma_length_3
    [0.01, 1]     # long_wma_alpha for EMA (second one)
]

# Wrapping the evaluate function to accept only weights
def fitness_function(weights):
    return evaluate(weights, prices)

# Initialize and run PSO
pso = ParticleSwarmOptimizer(
    fitness_function=fitness_function,
    bounds=bounds,
    num_particles=100,
    max_iter=100
)

best_params, best_score = pso.optimize()

print("Best Score: ", best_score)
print("Best Parameters: ")
print_parameters(best_params)

# Evaluate best parameters on test set
test_fitness = evaluate(best_params, testing_data[1], plot=True)
print(f"Test set fitness: {test_fitness:.2f}")
