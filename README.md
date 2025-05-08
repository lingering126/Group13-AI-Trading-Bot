# Bitcoin Trading Bot Optimization — CITS4404 Project (Group 13)

This repository contains the implementation of a trading bot that uses adaptive strategies to optimize Bitcoin trading performance based on historical data. It integrates nature-inspired optimization algorithms (PSO, ABC, HCA) to tune the parameters of a moving average crossover strategy.

# Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `AI_Project_Integrated_Code.ipynb` | **Main notebook**: This file integrates all components (data loading, bot logic, evaluation, optimization, plotting). Run this notebook to execute the entire experiment. |
| `ABC.py` | Implements the **Artificial Bee Colony (ABC)** optimization algorithm. |
| `PSO.py` | Implements the **Particle Swarm Optimization (PSO)** algorithm, with support for fitness tracking and iterative output. |
| `HCA.py` | Implements the **Hill Climbing Algorithm (HCA)** for local search-based parameter tuning. |
| `Call ABC` | Script that calls the ABC optimizer with appropriate parameters and configuration. |
| `Call PSO` | Script that calls the PSO optimizer. |
| `Call HCA` | Script that calls the HCA optimizer. |
| `Algorithm Comparison` | Contains code and results comparing algorithm performance on training and test datasets (e.g., final cash). |
| `bot.py` | Defines the trading bot evaluation logic, including moving average computation, buy/sell signal generation, and capital simulation. |
| `data/` | Contains historical BTC price data (`BTC-Daily.csv`) used as input for training and testing. |
| `legacy/` | Archive of earlier or alternative versions of evaluation functions and related code. |
| `.gitignore` | Specifies which files and folders to exclude from version control (e.g., `__pycache__`). |
| `README.md` | This file — provides project overview, file documentation, and usage instructions. |


# How to Run the Code

**Requirements:**  
- Jupyter Notebook or any Python environment that supports `.ipynb` files.
- Python 3.7+
- Standard scientific libraries: `numpy`, `pandas`, `matplotlib`

**Steps:**
1. Clone this repository.
2. Ensure the BTC price data file (`BTC-Daily.csv`) is placed inside the `data/` directory.
3. Open the file `AI_Project_Integrated_Code.ipynb`.
4. Run all cells from top to bottom. This includes:
   - Importing data
   - Constructing moving averages
   - Running optimization algorithms (PSO, ABC, HCA)
   - Visualizing trading signals
   - Comparing results

The final section of the notebook will generate training and testing performance plots and metrics for all three optimization algorithms.


# Project Scope and Objectives

This project was conducted as part of the **CITS4404 Artificial Intelligence Project** unit at The University of Western Australia. The aim was not just to produce a profitable trading bot, but to explore the following AI and optimization concepts:

- Understanding bots as pluggable components.
- Applying nature-inspired algorithms to generate and refine strategy instances.
- Using evaluation functions over real-world data to adaptively improve those strategies.
- Experimentally exploring the trade-offs between model flexibility, search space complexity, and algorithm performance.
- Comparing multiple optimization approaches through fair and controlled experiments.


# Algorithms Implemented

- **Particle Swarm Optimization (PSO)**  
  Swarm-based optimizer that balances personal and global experience to find high-performing solutions.

- **Artificial Bee Colony (ABC)**  
  Bee-foraging-inspired algorithm that emphasizes neighborhood search, probabilistic selection, and exploration through scout bees.

- **Hill Climbing Algorithm (HCA)**  
  Greedy local search algorithm used as a simple baseline for comparison.

Each optimizer is used to tune 14 continuous parameters of the bot, including moving average weights, window lengths, and smoothing factors.


# Contributors

This project was completed by Group 13 members for CITS4404:

- Eusha Khan (24245979)
- Joel Brooker (23409801) 
- Stefan Andonov (23374198)
- Wenbo Gao (23335934)
- Yi Ren (23895642)
- Zhi Wang (24560057) 
