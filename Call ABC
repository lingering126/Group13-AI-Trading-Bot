# ABC Invoke

# Main program
# Import data
data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

# Define parameter bounds
bounds = [
    (0, 1), (1, 100), (0, 1),  # SMA parameters
    (1, 100), (0, 1), (1, 100),  # LMA parameters
    (0, 1), (0, 1), (1, 100),  # EMA parameters
    (0, 1), (1, 100), (0, 1),  # SMA parameters
    (1, 100), (0, 1), (1, 100),  # LMA parameters
    (0, 1)  # EMA parameters
]


# Create objective function
def objective_function(params):
    return evaluate(params, training_data[1])


# Create ABC optimizer
abc = ABC(
    objective_func=objective_function,
    bounds=bounds,
    n_bees=50,
    n_iterations=100,
    limit=50
)

# Execute optimization
best_params, best_fitness = abc.optimize()

print("\nOptimization Results:")
print(f"Best parameters: {best_params}")
print(f"Best fitness: {best_fitness:.2f}")

# Evaluate best parameters on test set
test_fitness = evaluate(best_params, testing_data[1], plot=True)
print(f"Test set fitness: {test_fitness:.2f}")
