#HCA Invoke

# Import data
data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

# Define parameter bounds
bounds = [
            (0, 1), (1, 200), 
            (0, 1), (1, 200), 
            (0, 1), (1, 200), (0, 1), 
            (0, 1), (1, 200),
            (0, 1), (1, 200), 
            (0, 1), (1, 200), (0, 1)
        ]


# Create objective function
def objective_function(params):
    return evaluate(params, training_data[1])


# Create HCA optimizer
hca = HCA(
    objective_func=objective_function,
    bounds=bounds,
    step_size=0.1,
    n_iterations=1000
)

# Execute optimization
best_params, best_fitness = hca.optimize()

print("\nOptimization Results:")
print(f"Best parameters: {best_params}")
print(f"Best fitness: {best_fitness:.2f}")

# Evaluate best parameters on test set
test_fitness = evaluate(best_params, testing_data[1], plot=True)
print(f"Test set fitness: {test_fitness:.2f}")
