import random
import pickle
import os
import copy
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, search_space, elitism_count, mutation_rate,
                 checkpoint_path='ga_checkpoint.pkl'):
        self.population_size = population_size
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.population = []
        self.checkpoint_path = checkpoint_path

    def _create_random_chromosome(self):
        return {key: random.choice(values) for key, values in self.search_space.items()}

    def _coerce_gene_value(self, gene_name, value):
        valid_values = self.search_space[gene_name]
        if not valid_values:
            return value
        if value in valid_values:
            return value

        exemplar = valid_values[0]
        if isinstance(exemplar, (int, float, np.integer, np.floating)) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return min(valid_values, key=lambda candidate: abs(float(candidate) - float(value)))

        return random.choice(valid_values)

    def _sanitize_chromosome(self, chromosome):
        sanitized = chromosome.copy()
        for gene_name in self.search_space:
            if gene_name in sanitized:
                sanitized[gene_name] = self._coerce_gene_value(gene_name, sanitized[gene_name])
        return sanitized

    def _initialize_population(self):
        self.population = [{'chromosome': self._create_random_chromosome(), 'fitness': None} for _ in range(self.population_size)]

    def _save_checkpoint(self, generation_num):
        state = {'generation': generation_num, 'population': self.population}
        with open(self.checkpoint_path, 'wb') as f: pickle.dump(state, f)

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                population = state['population']
                for individual in population:
                    individual['chromosome'] = self._sanitize_chromosome(individual['chromosome'])
                return state['generation'], population
            except: pass
        return 0, None

    # # [ORIGINAL EVOLUTION STRATEGY — commented out 2026-05-24]
    # # # --- START LLM: CROSSOVER ---
    # def mutate_gene(self, current_value, possible_values):
    #     if not isinstance(possible_values, list):
    #         raise ValueError('possible_values should be a list')
    #     if not possible_values:
    #         return current_value
    #     import random
    #     new_value = random.choice(possible_values)
    #     if isinstance(current_value, (int, float, np.integer, np.floating)):
    #         if new_value == current_value:
    #             new_value += random.choice([-1, 1])
    #         else:
    #             new_value += random.choice([-0.1, 0.1])
    #     return new_value
    # def _mutate(self, chromosome):
    #     ...
    # def combine_genes(self, gene_name, ...):
    #     ...  # Had 'nn_blocks' typo
    # def _crossover(self, parent1_chromo, parent2_chromo):
    #     ...
    # def select_competitor(self, competitors):
    #     ...  # No ZeroDivisionError guard
    # def _selection(self):
    #     ...
    # # [END ORIGINAL]

    # --- START LLM: EVOLUTION STRATEGY ---
    def combine_genes(self, gene_name, parent1_value, parent2_value, crossover_point, gene_index, total_genes):
            """Decide which parent's gene to use for a child chromosome."""
            if parent1_value == parent2_value:
                return parent1_value
            if gene_name in ['lr', 'momentum']:
                return random.choice([parent1_value, parent2_value])
            elif gene_name in ['n_columns', 'base_channels', 'dropout_prob', 'n_blocks']:
                return random.choice([parent1_value, parent2_value])
            else:
                return random.choice([parent1_value, parent2_value])

    def _crossover(self, parent1_chromo, parent2_chromo):
        child_chromo = {}
        genes = list(self.search_space.keys())
        point = random.randint(1, len(genes) - 1)
        for i, gene in enumerate(genes):
            child_chromo[gene] = self._coerce_gene_value(
                gene,
                self.combine_genes(gene, parent1_chromo[gene], parent2_chromo[gene], point, i, len(genes)),
            )
        return self._sanitize_chromosome(child_chromo)

    def mutate_gene(self, current_value, possible_values):
        if not isinstance(possible_values, list):
            raise ValueError('possible_values should be a list')
        if not possible_values:
            return current_value
        # Strictly pick from possible_values to prevent out-of-bounds errors
        if isinstance(current_value, (int, float, np.integer, np.floating)):
            # Simulate random walk by picking nearest valid neighbor
            return min(possible_values, key=lambda x: abs(float(x) - float(current_value)))
        return random.choice(possible_values)

    def _mutate(self, chromosome):
        mutated_chromo = chromosome.copy()
        for gene in self.search_space.keys():
            if random.random() < self.mutation_rate:
                possibles = [v for v in self.search_space[gene] if v != mutated_chromo[gene]]
                if possibles:
                    mutated_chromo[gene] = self._coerce_gene_value(
                        gene,
                        self.mutate_gene(mutated_chromo[gene], possibles)
                    )
        return self._sanitize_chromosome(mutated_chromo)

    def select_competitor(self, competitors):
        fitnesses = [(x['fitness'] if x['fitness'] is not None else 0) for x in competitors]
        total_fitness = sum(fitnesses)
        if total_fitness <= 0:
            return random.choice(competitors)
        probabilities = [f / total_fitness for f in fitnesses]
        return random.choices(competitors, weights=probabilities, k=1)[0]

    def _selection(self):
        k = 3
        competitors = random.sample(self.population, min(k, len(self.population)))
        return self.select_competitor(competitors)
    # --- END LLM: EVOLUTION STRATEGY ---

    def run(self, num_generations, fitness_function):
        start_gen, loaded_population = self._load_checkpoint()
        if loaded_population is not None: self.population = loaded_population
        else: self._initialize_population()
            
        fitness_history = []
        best_overall = None

        for gen in range(start_gen, num_generations):
            print(f"\n\n >>> GENERATION {gen + 1} <<<\n")
            # Evaluate
            for i, ind in enumerate(self.population):
                ind['chromosome'] = self._sanitize_chromosome(ind['chromosome'])
                if ind['fitness'] is None:
                    print(f"  Evaluating Individual {i+1}/{len(self.population)} ---- Generation: {gen+1}/{num_generations}")
                    ind['fitness'] = fitness_function(ind['chromosome'])
            
            # Sort
            self.population.sort(key=lambda x: x['fitness'] if x['fitness'] is not None else -1, reverse=True)
            
            # Record keeping
            current_best = self.population[0]['fitness']
            fitness_history.append(current_best)
            if best_overall is None or current_best > best_overall['fitness']:
                best_overall = self.population[0].copy()

            # Next Gen (Using deepcopy to protect elites from accidental mutation)
            # next_gen = self.population[:self.elitism_count]
            next_gen = copy.deepcopy(self.population[:self.elitism_count])
            while len(next_gen) < self.population_size:
                p1 = self._selection()
                p2 = self._selection()
                child = self._crossover(p1['chromosome'], p2['chromosome'])
                child = self._mutate(child)
                next_gen.append({'chromosome': child, 'fitness': None})
            
            self.population = next_gen
            self._save_checkpoint(gen + 1)
            
        return best_overall, fitness_history
