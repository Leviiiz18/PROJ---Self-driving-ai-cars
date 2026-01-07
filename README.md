# NEAT-Based Racing Simulation

This project is a racing simulation built using **NEAT (NeuroEvolution of Augmenting Topologies)** in Python.  
The goal is to train AI agents to learn how to race and improve their performance over generations using evolutionary algorithms instead of hard-coded rules.

The project uses neural networks that evolve over time based on fitness scores from the racing environment.

---

## Files in the Project

- `p.py`  
  Main simulation file. Runs the racing environment and controls the training loop.

- `m.py`  
  Helper module for agent logic, movement, or environment handling.

- `m2.py`  
  Additional helper or experimental module used for testing or extensions.

- `race_monitor.py`  
  Monitors race progress, logs performance, and helps track training results.

- `neat-config.txt` / `neat_config.txt`  
  Configuration files for NEAT.  
  These define population size, mutation rate, crossover rules, activation functions, etc.  
  (Only one is needed — the duplicate can be removed.)

- `racing_simulation_20250711_112633.log`  
  Log file containing training or simulation output for debugging and analysis.

---

## How the Project Works

1. NEAT initializes a population of neural networks.
2. Each network controls a racing agent.
3. Agents race in the simulation environment.
4. Performance is measured using a fitness function.
5. Better-performing agents are selected.
6. New generations are created using mutation and crossover.
7. Over time, agents learn better racing behavior automatically.

No reinforcement learning libraries are used — learning happens purely through evolution.

---

## Requirements

- Python 3.8+
- neat-python
- pygame (if the simulation is visual)

Install dependencies using:
```bash
pip install neat-python pygame
