import pandas as pd
from datetime import datetime
import time

def run_all_optimizers(data_path, n_runs=5):
    results = []

    for run in range(1, n_runs + 1):
        print(f"\n=== Run {run} ===")
        data = import_data(data_path)
        training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())
        prices = training_data[1]

        # PSO
        pso_bounds = [
            [0, 1], [1, 200], 
            [0, 1], [1, 200],
            [0, 1], [1, 200], [0.01, 1],
            [0, 1], [1, 200], 
            [0, 1], [1, 200],
            [0, 1], [1, 200], [0.01, 1]
        ]

        start_time = time.time()
        pso = ParticleSwarmOptimizer(
            fitness_function=lambda w: evaluate(w, prices),
            bounds=pso_bounds,
            num_particles=100,
            max_iter=100
        )
        pso_best_params, pso_best_train = pso.optimize()
        pso_test_score = evaluate(pso_best_params, testing_data[1], plot=False)
        end_time = time.time()
        results.append({
            "Run": run,
            "Algorithm": "PSO",
            "Train Score": pso_best_train,
            "Test Score": pso_test_score,
            "Time (s)": round(end_time - start_time, 2)
        })

        # HCA
        hca_bounds = [
            (0, 1), (1, 200), 
            (0, 1), (1, 200), 
            (0, 1), (1, 200), (0, 1), 
            (0, 1), (1, 200),
            (0, 1), (1, 200), 
            (0, 1), (1, 200), (0, 1)
        ]
        
        start_time = time.time()
        hca = HCA(
            objective_func=lambda p: evaluate(p, training_data[1]),
            bounds=hca_bounds,
            step_size=0.1,
            n_iterations=1000
        )
        hca_best_params, hca_best_train = hca.optimize()
        hca_test_score = evaluate(hca_best_params, testing_data[1], plot=False)
        end_time = time.time()
        results.append({
            "Run": run,
            "Algorithm": "HCA",
            "Train Score": hca_best_train,
            "Test Score": hca_test_score,
            "Time (s)": round(end_time - start_time, 2)
        })

        # ABC
        start_time = time.time()
        abc = ABC(
            objective_func=lambda p: evaluate(p, training_data[1]),
            bounds=hca_bounds,
            n_bees=50,
            n_iterations=100,
            limit=50
        )
        abc_best_params, abc_best_train = abc.optimize()
        abc_test_score = evaluate(abc_best_params, testing_data[1], plot=False)
        end_time = time.time()
        results.append({
            "Run": run,
            "Algorithm": "ABC",
            "Train Score": abc_best_train,
            "Test Score": abc_test_score,
            "Time (s)": round(end_time - start_time, 2)
        })

    df = pd.DataFrame(results)
    summary = df.groupby("Algorithm").agg({
        "Train Score": ["mean", lambda x: max(x) - min(x)],
        "Test Score": ["mean", lambda x: max(x) - min(x)],
        "Time (s)": "mean"
    })
    summary.columns = ["Train Avg", "Train Range", "Test Avg", "Test Range", "Avg Time (s)"]
    return df, summary

results_df, summary_df = run_all_optimizers(DATA_PATH, n_runs=5)
print(results_df)
print("\nSummary:\n", summary_df)
