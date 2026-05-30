import os
import re
import ast
import random
import subprocess
import textwrap
import shutil
import time
import json
import sys
from datetime import datetime

from ab.gpt.brute.ga.meta_evolution.llm_loader import LocalLLMLoader 
from ab.gpt.brute.ga.meta_evolution.rl_rewards import calculate_meta_reward
from ab.gpt.brute.ga.meta_evolution.FractalNet_evolvable import SEARCH_SPACE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(BASE_DIR, "genetic_algorithm.py")
RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_fractal_evolution.py")
BACKUP_DIR = os.path.join(BASE_DIR, "ga_history_backup")
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"LLM-evolution-logs_{RUN_TIMESTAMP}.jsonl")
GA_EVAL_LOG_FILE = os.path.join(LOGS_DIR, f"ga_evaluations_{RUN_TIMESTAMP}.jsonl")
ADAPTER_SAVE_PATH = os.path.join(BASE_DIR, "fine_tuned_adapter")

# KEEP BENCHMARKS SMALL FOR FAST FEEDBACK, BUT CONFIGURABLE VIA ENV
BENCH_GENS = int(os.environ.get("GENERATIONS", 3))
BENCH_POP = int(os.environ.get("POPULATION_SIZE", 10)) 

# --- FULL CONTEXT PROMPT TEMPLATE ---
# Format the search space so the LLM understands the exact gene constraints
SEARCH_SPACE_STR = json.dumps(SEARCH_SPACE, indent=2)

# # [ORIGINAL BASE_PROMPT_TEMPLATE — commented out 2026-05-24]
# BASE_PROMPT_TEMPLATE = """
# You are an expert AI researcher fine-tuning a Genetic Algorithm (GA)...
# === SEARCH SPACE === ... === STRICT OUTPUT FORMAT ===
# 1. Output ONLY the python code...
# """
# # [END ORIGINAL]

BASE_PROMPT_TEMPLATE = """
### ROLE & OBJECTIVE ###
You are an elite AI Research Engineer and Evolutionary Computation Expert. 
Your task is to rewrite the core evolutionary operators of a Quality-Diversity Genetic Algorithm (MAP-Elites) that optimizes PyTorch Neural Network architectures (FractalNet) for CIFAR-10.
Your generated code will be AUTOMATICALLY INJECTED into a production Python class via string replacement. Any syntax error, hallucinated import, or invalid indentation will crash the entire pipeline.

Your objective is NOT just to produce a "good" algorithm in the abstract. You must propose crossover and mutation strategies that will theoretically expand the frontier of what the GA has seen:
1. Break the Global SOTA peak accuracy.
2. Discover novel, high-performing architectures to populate empty cells in the MAP-Elites Behavioral Archive.

### HARD CONSTRAINTS (CRITICAL) ###
1. NO INVENTED VALUES: Genes are strictly discrete. You MUST ONLY select values from the provided `SEARCH_SPACE` or the `possible_values` list. NEVER use arithmetic (e.g., `val + 0.1`, `val * 2`, `random.uniform()`) to create new gene values.
2. NO IMPORTS: Do NOT write `import random` or `import numpy as np`. Assume `random`, `numpy as np`, and `copy` are already available in the global scope.
3. EXACT SIGNATURES: You must use the EXACT method signatures provided. Do not add or remove arguments.
4. CLASS INDENTATION: Your output will be injected directly into the `GeneticAlgorithm` class. All `def` statements MUST have exactly 4 spaces of indentation. Method bodies MUST have 8 spaces.
5. SANITIZATION: Always pass newly created chromosomes through `self._sanitize_chromosome(child_chromo)` before returning them in `_crossover` and `_mutate`.

### ANTI-PATTERNS (NEVER DO THIS) ###
❌ BAD: `new_lr = current_value + 0.001` (Invents a value not in search space)
✅ GOOD: `new_lr = min(possible_values, key=lambda x: abs(x - current_value))` (Snaps to nearest valid value)
❌ BAD: `import random` inside the method (Causes IndentationError/SyntaxError upon injection)
✅ GOOD: Just use `random.choice()` directly.
❌ BAD: `total_fitness = sum(...); prob = fit / total_fitness` (Crashes if all fitnesses are 0)
✅ GOOD: Add a fallback: `if total_fitness <= 0: return random.choice(competitors)`

=== SEARCH SPACE ===
The GA optimizes the following discrete SEARCH_SPACE:
{search_space}

=== FULL CLASS CONTEXT ===
Here is the current full implementation of the GeneticAlgorithm class for your reference.
Notice how the GA uses a MAP-Elites archive based on `n_blocks` and `base_channels`.
<full_code>
{full_code}
</full_code>

=== CURRENT SOTA & ARCHIVE FRONTIER ===
- All-Time Global Best Peak Accuracy (SOTA): {global_best_score:.2f}%
- MAP-Elites Archive Size: {global_archive_size} unique cells discovered

=== PREVIOUS ATTEMPTS & FEEDBACK ===
Learn from past failures. DO NOT repeat failed logic. If a previous attempt failed, try a fundamentally different mathematical or probabilistic approach.
{history_str}

=== YOUR SPECIFIC TASK ===
You must rewrite the following component: `{method_name}`
{task_specific_instructions}

Current implementation to be replaced:
<current_function>
{code}
</current_function>

=== STRICT OUTPUT FORMAT ===
1. Wrap your ENTIRE Python code output in a single ```python ... ``` markdown block.
2. DO NOT write ANY text, explanations, thinking, or comments before or after the markdown block.
3. The code inside the block MUST start exactly with `    def ` (4 spaces indent).
4. Output EXACTLY ONE method: `{method_name}`.
"""

INSTRUCTIONS = {
    "combine_genes": "Task: Implement `combine_genes` to decide which parent's gene to use for a child chromosome. For numeric genes (lr, momentum), prefer higher/lower or random choice. For categorical, random choice is safe.",
    "_crossover": "Task: Implement `_crossover`. Return a new chromosome dict by crossing over parent1_chromo and parent2_chromo using combine_genes. You MUST return self._sanitize_chromosome(child_chromo).",
    "mutate_gene": "Task: Implement `mutate_gene`. You are given `possible_values`. Pick one. For numeric genes, picking the nearest valid value is good. For categorical, use random.choice.",
    "_mutate": "Task: Implement `_mutate`. Return a new chromosome dict with mutated genes based on self.mutation_rate. You MUST return self._sanitize_chromosome(mutated_chromo).",
    "select_competitor": "Task: Implement `select_competitor`. Return a single competitor from the `competitors` list. Handle the edge case where all fitnesses are 0 to prevent ZeroDivisionError.",
    "_selection": "Task: Implement `_selection`. Select a pool of competitors from self.population and pass them to select_competitor. Return the chosen competitor."
}

class MetaEvolver:
    def __init__(self, model_path):
        self.llm = LocalLLMLoader(model_path, use_quantization=True, adapter_path=ADAPTER_SAVE_PATH)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        # print("[Meta] Running Baseline...")
        # self.baseline_score = self.run_benchmark()
        # print(f"[Meta] Baseline Score: {self.baseline_score:.4f}")
        self.baseline_score = 0.0
        self.global_best_score = 0.0
        self.global_archive_size = 0
        
        # --- Experience Replay Buffer ---
        self.success_buffer = []
        self.attempt_history = []

    def run_benchmark(self):
        import statistics
        # Removed --clean to persist the GA archive and population across LLM attempts
        cmd = [sys.executable, RUNNER_SCRIPT, "--gens", str(BENCH_GENS), "--pop", str(BENCH_POP)]
        env = os.environ.copy()
        env["GA_EVAL_LOG"] = GA_EVAL_LOG_FILE
        
        runs = 3
        scores = []
        peak_accs = []
        
        for i in range(runs):
            print(f"\n[Meta] Running Benchmark {i+1}/{runs}...")
            try:
                # Real-Time Logging with Popen
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                
                full_output = ""
                for line in process.stdout:
                    print(line, end="") # Stream line
                    full_output += line
                    
                process.wait(timeout=21600)
                
                score_match = re.search(r"META_SCORE:\s*([\d\.]+)", full_output)
                score = float(score_match.group(1)) if score_match else 0.0
                
                peak_match = re.search(r"PEAK_ACCURACY:\s*([\d\.]+)", full_output)
                peak_acc = float(peak_match.group(1)) if peak_match else 0.0

                top3_match = re.search(r"TOP3_MEAN:\s*([\d\.]+)", full_output)
                top3_acc = float(top3_match.group(1)) if top3_match else peak_acc
                
                archive_match = re.search(r"ARCHIVE_SIZE:\s*(\d+)", full_output)
                archive_size = int(archive_match.group(1)) if archive_match else 0
                
                if not top3_match:
                    print(f"[Meta] Benchmark Output (Snippet):\n{full_output[-1000:]}")
                
                scores.append(top3_acc) # Track Top3 for secondary density reward
                peak_accs.append(peak_acc)
            except Exception as e: 
                 print(f"[Meta] Benchmark Exception: {e}")
                 scores.append(0.0)
                 peak_accs.append(0.0)
                 archive_size = 0
        
        median_top3 = statistics.median(scores) if scores else 0.0
        median_peak = statistics.median(peak_accs) if peak_accs else 0.0
        print(f"[Meta] Median Top-3 Quality over {runs} runs: {median_top3:.4f}")
        return {"top3_mean": median_top3, "peak_accuracy": median_peak, "archive_size": archive_size}

    def _extract_method(self, source_code, method_name):
        import ast
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == 'GeneticAlgorithm':
                    for body_item in node.body:
                        if isinstance(body_item, ast.FunctionDef) and body_item.name == method_name:
                            lines = source_code.splitlines()
                            start_line = body_item.lineno - 1
                            end_line = body_item.end_lineno
                            
                            if body_item.decorator_list:
                                start_line = body_item.decorator_list[0].lineno - 1

                            char_start = sum(len(l) + 1 for l in lines[:start_line])
                            char_end = sum(len(l) + 1 for l in lines[:end_line])
                            extracted_source = "\n".join(lines[start_line:end_line]) + "\n"
                            
                            return extracted_source, (char_start, char_end), 4
        except Exception as e:
            print(f"[Meta] AST extraction failed: {e}")
        return None, None, 0

    def _extract_function_body(self, text, method_name):
        """Extract the code block from the LLM response robustly."""
        if any(bad in text for bad in ["import torch", "class FractalNet", "def train("]):
            print(f"[Meta] Validation failed: LLM generated hallucinated codebase.")
            return None
            
        import re
        blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
            
        if blocks:
            best_block = None
            best_score = -1
            for block in blocks:
                block_stripped = block.strip()
                score = 0
                if f"def {method_name}" in block_stripped: score += 10
                if "pass" in block_stripped or "..." in block_stripped: score -= 5
                if len(block_stripped) > 20: score += 1
                if score > best_score:
                    best_score = score
                    best_block = block_stripped
            if best_block: return best_block
            
        # Ultimate fallback: strip conversational text
        lines = text.strip().splitlines()
        code_lines = [l for l in lines if not l.strip().startswith(("Here", "Note", "I have", "```"))]
        return "\n".join(code_lines).strip()

    def evolve_component(self, method_name, attempt=1, total_attempts=5):
        print(f"\n[Meta] Evolving: {method_name}")
        with open(TARGET_FILE, 'r') as f: full_code = f.read()
        
        orig_code, span, indent_col = self._extract_method(full_code, method_name)
        if not orig_code: return

        print(f"--- [DEBUG] Input Code ---\n{orig_code}\n-------------------------")

        # Format history string
        history_str = "No previous attempts yet."
        if hasattr(self, 'attempt_history') and self.attempt_history:
            history_lines = []
            for idx, h in enumerate(self.attempt_history[-3:]):
                status = h.get("status", "Unknown")
                score = h.get("score", 0.0)
                history_lines.append(f"Attempt {idx+1}:\nStatus: {status}\nScore: {score}\nCode:\n```python\n{h.get('code')}\n```")
            history_str = "\n\n".join(history_lines)

        # LLM Generation with Full Context
        prompt = BASE_PROMPT_TEMPLATE.format(
            search_space=SEARCH_SPACE_STR,
            full_code=full_code,
            method_name=method_name,
            task_specific_instructions=INSTRUCTIONS.get(method_name, ""),
            code=orig_code,
            history_str=history_str,
            global_best_score=self.global_best_score,
            global_archive_size=self.global_archive_size
        )
        
        # Temperature scheduling: 0.9 down to 0.4
        progress = (attempt - 1) / max(1, total_attempts - 1)
        temperature = max(0.4, 0.9 - (0.5 * progress))
        print(f"[Meta] Generation Temperature: {temperature:.2f}")
        
        raw_res = self.llm.generate(prompt, max_new_tokens=2048, temperature=temperature)
        
        print(f"--- [DEBUG] Raw Response ---\n{raw_res}\n--------------------------")
        
        # Cleanup & Extraction
        extracted = self._extract_function_body(raw_res, method_name)
        
        if extracted:
            new_code = extracted
        else:
            print(f"[Meta] Failed to extract valid {method_name} from LLM response. Skipping.")
            return

        # Normalize Indentation
        new_code = textwrap.dedent(new_code).strip()
        
        print(f"--- [DEBUG] Cleaned Code (Normalized) ---\n{new_code}\n---------------------------")
        
        # Indentation for Injection
        reindented = "\n".join([" " * indent_col + line for line in new_code.splitlines()])

        # Syntax Check & Injection
        valid_syntax = False
        try:
            test_full = full_code[:span[0]] + reindented + "\n" + full_code[span[1]:]

            # DEBUG: Print what we are trying to inject
            print(f"--- [DEBUG] Injection Snippet ---\n{test_full[span[0]-50:span[0]+len(reindented)+50]}\n-------------------------------")

            ast.parse(test_full)
            valid_syntax = True
        except SyntaxError as e:
            print(f"[Meta] Syntax Error: {e}")

        # new_score = 0.0
        bench_stats = {"top3_mean": 0.0, "peak_accuracy": 0.0, "archive_size": self.global_archive_size}
        
        if valid_syntax:
            target_filename = os.path.basename(TARGET_FILE)
            bkp = os.path.join(BACKUP_DIR, f"{target_filename}_{method_name}.bak")
            shutil.copy(TARGET_FILE, bkp)
            with open(TARGET_FILE, 'w') as f: f.write(test_full)
            
            # --- Runtime Smoke Test: catch NameError/TypeError before expensive benchmark ---
            try:
                import importlib
                import ab.gpt.brute.ga.meta_evolution.genetic_algorithm as ga_mod
                importlib.reload(ga_mod)
                test_ga = ga_mod.GeneticAlgorithm(
                    population_size=4, search_space=SEARCH_SPACE,
                    elitism_count=1, mutation_rate=1.0,
                    checkpoint_path="/dev/null"
                )
                # Create a test chromosome and exercise all LLM-editable functions
                test_chromo = test_ga._create_random_chromosome()
                test_chromo2 = test_ga._create_random_chromosome()
                mutated = test_ga._mutate(test_chromo)
                crossed = test_ga._crossover(test_chromo, test_chromo2)
                # Validate mutated values are within search space
                for gene, val in mutated.items():
                    if val not in SEARCH_SPACE[gene]:
                        raise ValueError(f"Smoke test: mutate produced '{val}' for gene '{gene}', not in search space {SEARCH_SPACE[gene]}")
                for gene, val in crossed.items():
                    if val not in SEARCH_SPACE[gene]:
                        raise ValueError(f"Smoke test: crossover produced '{val}' for gene '{gene}', not in search space {SEARCH_SPACE[gene]}")
                print("[Meta] Runtime smoke test PASSED.")
            except Exception as e:
                print(f"[Meta] Runtime smoke test FAILED: {e}")
                print("---> Reverting file and skipping benchmark.")
                shutil.copy(bkp, TARGET_FILE)
                valid_syntax = False
            
            if valid_syntax:
                print("[Meta] Benchmarking...")
                bench_stats = self.run_benchmark()

        new_score = bench_stats["peak_accuracy"]
        top3_mean = bench_stats["top3_mean"]
        new_archive_size = bench_stats["archive_size"]
        
        # Calculate novelty (how many new cells were filled)
        archive_novelty = max(0, new_archive_size - self.global_archive_size)

        # RL Loop: Pass all frontier metrics to the reward calculator
        reward = calculate_meta_reward(
            current_score=new_score, 
            best_ever_score=self.global_best_score, 
            top3_mean=top3_mean, 
            archive_novelty=archive_novelty, 
            valid_syntax=valid_syntax
        )
        
        # Update Global Frontier Tracking
        if valid_syntax:
            if new_score > self.global_best_score:
                print(f"[Meta] --> NEW GLOBAL SOTA! {new_score:.2f}% (was {self.global_best_score:.2f}%)")
                self.global_best_score = new_score
            if new_archive_size > self.global_archive_size:
                print(f"[Meta] --> ARCHIVE EXPANDED! {new_archive_size} cells (was {self.global_archive_size})")
                self.global_archive_size = new_archive_size
        
        fine_tune_expected = True # We always train now! (Either positive or negative)
        fine_tune_started = False
        fine_tune_completed = False
        fine_tune_failed = False
        fine_tune_exception = None
        adapter_save_started = False
        adapter_save_completed = False
        adapter_save_failed = False
        adapter_save_exception = None
        adapter_path = ADAPTER_SAVE_PATH
        train_examples_count = 1 if fine_tune_expected else 0
        train_epochs = 1 if fine_tune_expected else 0
        fine_tune_start_time = None
        fine_tune_end_time = None
        adapter_save_start_time = None
        adapter_save_end_time = None
        
        if valid_syntax and reward > 0:
            print("--> SUCCESS. Updating Baseline (EMA) & Fine-tuning on Success.")
            # EMA Update: 20% of the new score, 80% of the old baseline
            if self.baseline_score == 0.0:
                self.baseline_score = new_score  # Initialize on first success
            else:
                self.baseline_score = (0.2 * new_score) + (0.8 * self.baseline_score)
            
            # --- Experience Replay: append success ---
            self.success_buffer.append({'prompt': prompt, 'completion': new_code})
            if len(self.success_buffer) > 20:
                self.success_buffer.pop(0)  # Cap buffer at 20, drop oldest
        else:
            print("--> FAILURE/REGRESSION. Fine-tuning on Baseline (Negative Sampling).")
            # If valid_syntax is false or reward <= 0, we teach it to revert to orig_code
            self.success_buffer.append({'prompt': prompt, 'completion': orig_code})
            if len(self.success_buffer) > 20:
                self.success_buffer.pop(0)

        # Sample a batch
        batch_size = min(4, len(self.success_buffer))
        training_batch = random.sample(self.success_buffer, batch_size)
        train_examples_count = batch_size
        
        print(f"[LoRA] Fine-tune start (replay buffer: {len(self.success_buffer)} items, batch: {batch_size})")
        fine_tune_started = True
        fine_tune_start_time = datetime.now().isoformat()
        try:
            self.llm.train_on_buffer(training_batch, epochs=train_epochs)
            fine_tune_completed = True
            print("[LoRA] Fine-tune complete")
        except Exception as e:
            fine_tune_failed = True
            fine_tune_exception = str(e)
            print(f"[LoRA] Fine-tune failed: {e}")
        finally:
            fine_tune_end_time = datetime.now().isoformat()
            
        if fine_tune_completed:
            print("[LoRA] Adapter save start")
            adapter_save_started = True
            adapter_save_start_time = datetime.now().isoformat()
            try:
                self.llm.save_adapters(ADAPTER_SAVE_PATH)
                adapter_save_completed = True
                print("[LoRA] Adapter save complete")
            except Exception as e:
                adapter_save_failed = True
                adapter_save_exception = str(e)
                print(f"[LoRA] Adapter save failed: {e}")
            finally:
                adapter_save_end_time = datetime.now().isoformat()
                
        if 'bkp' in locals() and not (valid_syntax and reward > 0):
            print("--> Reverting File.")
            shutil.copy(bkp, TARGET_FILE)
            
        # Log attempt history
        status = "Success" if (valid_syntax and reward > 0) else ("Syntax Error" if not valid_syntax else "Regression")
        if not hasattr(self, 'attempt_history'): self.attempt_history = []
        self.attempt_history.append({"status": status, "score": new_score, "code": new_code})
            
        # Log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "prompt": prompt,
            "response": raw_res,
            "cleaned_code": new_code,
            "valid_syntax": valid_syntax,
            "score": new_score,
            "peak_accuracy": bench_stats["peak_accuracy"],
            "reward": reward,
            "fine_tune_expected": fine_tune_expected,
            "fine_tune_started": fine_tune_started,
            "fine_tune_completed": fine_tune_completed,
            "fine_tune_failed": fine_tune_failed,
            "fine_tune_exception": fine_tune_exception,
            "adapter_save_started": adapter_save_started,
            "adapter_save_completed": adapter_save_completed,
            "adapter_save_failed": adapter_save_failed,
            "adapter_save_exception": adapter_save_exception,
            "adapter_path": adapter_path,
            "train_examples_count": train_examples_count,
            "train_epochs": train_epochs,
            "fine_tune_start_time": fine_tune_start_time,
            "fine_tune_end_time": fine_tune_end_time,
            "adapter_save_start_time": adapter_save_start_time,
            "adapter_save_end_time": adapter_save_end_time
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

        # Return True if this attempt was a success (valid + improved)
        return bool(valid_syntax and reward > 0)

if __name__ == "__main__":
    # MODEL_PATH = "deepseek-ai/deepseek-coder-6.7b-instruct" 
    MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
    evolver = MetaEvolver(MODEL_PATH)

    # LOOP: CYCLE THROUGH ALL EVOLVABLE COMPONENTS
    # [OLD — fixed attempt count, stopped after N tries regardless of success]
    # META_ITERATIONS = int(os.environ.get("META_ATTEMPTS"))
    # for i in range(META_ITERATIONS):
    #     component = COMPONENTS[i % len(COMPONENTS)]
    #     print(f"\n=== Meta-Evolution Iteration {i+1}/{META_ITERATIONS} ===")
    #     evolver.evolve_component(component, attempt=i+1, total_attempts=META_ITERATIONS)
    #     time.sleep(2)
    # [END OLD]

    # NEW — keep trying until META_ITERATIONS *successful* evolutions are achieved
    # Priority: env var META_ATTEMPTS > model_config.json meta_attempts > default 5
    META_ITERATIONS = int(os.environ.get("META_ATTEMPTS", evolver.llm.config.get("meta_attempts", 5)))
    COMPONENTS = ["combine_genes", "_crossover", "mutate_gene", "_mutate", "select_competitor", "_selection"]

    successes = 0
    total_attempts = 0
    print(f"\n[Meta] Target: {META_ITERATIONS} SUCCESSFUL evolutions (will retry on failure)")
    while successes < META_ITERATIONS:
        total_attempts += 1
        component = COMPONENTS[(total_attempts - 1) % len(COMPONENTS)]
        print(f"\n=== Attempt {total_attempts} | Successes: {successes}/{META_ITERATIONS} — Evolving: {component} ===")
        success = evolver.evolve_component(component, attempt=total_attempts, total_attempts=META_ITERATIONS)
        if success:
            successes += 1
            print(f"[Meta] ✓ Success #{successes}/{META_ITERATIONS} achieved on attempt {total_attempts}")
        else:
            print(f"[Meta] ✗ Attempt {total_attempts} failed — retrying (successes so far: {successes}/{META_ITERATIONS})")
        time.sleep(2)
    print(f"\n=== Meta-Evolution Complete: {successes} successful evolutions in {total_attempts} total attempts ===")
    
    # --- Generate visualizations after all iterations ---
    try:
        from ab.gpt.brute.ga.meta_evolution.visualize_training import main as generate_plots
        print("\n=== Generating Visualizations ===")
        generate_plots()
    except Exception as e:
        print(f"[WARN] Visualization failed (non-fatal): {e}")
