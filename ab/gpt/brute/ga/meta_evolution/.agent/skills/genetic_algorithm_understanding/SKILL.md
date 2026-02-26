---
name: Genetic Algorithm Understanding
description: Guide and tools for understanding the Genetic Algorithm (GA) implementation used in this project.
---

# Instructions

This skill focuses on the core Genetic Algorithm logic found in `ab/gpt/brute/ga/modular/genetic_algorithm.py`.

1. **Core Loop**:
   - The GA typically involves: **Selection**, **Crossover**, **Mutation**, and **Replacement**.
   - Look for the `evolve()` or `step()` function as the main driver.

2. **Operators**:
   - **Selection**: Identified by functions like `select_parents`, `tournament_selection`, etc.
   - **Crossover**: Look for `crossover`, `mate`, or recombine logic.
   - **Mutation**: Look for `mutate`. This is often where random changes are introduced to the genome.

3. **Fitness Evaluation**:
   - This connects the GA to the problem domain (Fractal Networks).
   - The fitness function receives a genome and returns a score.
   - Look for where the GA calls out to an external evaluator or uses a direct objective function.

4. **Parameters**:
   - Population Size: How many individuals are maintained.
   - Mutation Rate: Probability of a mutation occurring.
   - Crossover Rate: Probability of crossover occurring.
   - Elitism: Number of best individuals to keep unchanged.

5. **Usage**:
   - Use this skill when tuning GA parameters or modifying the evolutionary operators.
   - If asked to "fix convergence issues", check the Selection pressure and Mutation rates here.
