"""
visualize_meta_generation.py
---------------------
Generates and saves plots for:
  1. GA evolution progress  (from stats/ JSON files)
  2. LLM fine-tuning progress (from LLM-evolution-logs.jsonl)

Output is saved into a timestamped folder:
  meta_evolution/visualizations/run_<YYYY-MM-DD_HH-MM-SS>/
      ga_evolution/
          generation_accuracy.png
          population_diversity.png
          best_vs_avg_accuracy.png
      fine_tuning/
          reward_over_iterations.png
          syntax_success_rate.png
          score_improvement.png

Usage:
    python3 visualize_meta_generation.py
"""

import os
import json
import warnings
import glob
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
STATS_DIR      = os.path.join(BASE_DIR, "stats")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
VIZ_ROOT       = os.path.join(BASE_DIR, "visualizations")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# PLOT_STYLE = {
#     "figure.facecolor": "#0f1117",
#     "axes.facecolor":   "#1a1d2e",
#     "axes.edgecolor":   "#3a3f5c",
#     "axes.labelcolor":  "#e0e0e0",
#     "xtick.color":      "#b0b0b0",
#     "ytick.color":      "#b0b0b0",
#     "text.color":       "#e0e0e0",
#     "grid.color":       "#2a2d3e",
#     "legend.facecolor": "#1a1d2e",
#     "legend.edgecolor": "#3a3f5c",
# }
PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "black",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "text.color":       "black",
    "grid.color":       "gray",
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
}

ACCENT1 = "#7c83fd"   # blue-purple
ACCENT2 = "#fd7c83"   # coral
ACCENT3 = "#7cfd83"   # green
BAR_COLOR = "#3a4a8a"


def _apply_style(ax, title, xlabel, ylabel):
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    # ax.grid(True, linestyle="--", alpha=0.4)
    # ax.tick_params(colors="#b0b0b0")
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(colors="black")


def _save(fig, path, saved_files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor='white', transparent=False)
    plt.close(fig)
    saved_files.append(path)
    print(f"  [saved] {os.path.relpath(path, BASE_DIR)}")


def _warn(msg):
    print(f"  [WARN]  {msg}")

def _extract_log_timestamp(target_ts=None):
    """
    Extract the experiment timestamp from source log filenames.
    Priority: target_ts > ga_evaluations > LLM-evolution-logs > pod log > fallback to now().
    """
    if target_ts:
        target_files = glob.glob(os.path.join(BASE_DIR, "logs*", f"*{target_ts}*.jsonl"))
        is_cifar100 = any("cifar100" in os.path.basename(f) for f in target_files)
        return target_ts, is_cifar100

    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')
    
    # Priority order of log file patterns to check
    search_patterns = [
        os.path.join(BASE_DIR, "logs_cifar10", "ga_evaluations_*.jsonl"),
        os.path.join(BASE_DIR, "logs_cifar100", "ga_evaluations_*.jsonl"),
        os.path.join(BASE_DIR, "logs_cifar10", "LLM-evolution-logs_*.jsonl"),
        os.path.join(BASE_DIR, "logs_cifar100", "LLM-evolution-logs_*.jsonl"),
        os.path.join(LOGS_DIR, "ga_evaluations_*.jsonl"),
        os.path.join(LOGS_DIR, "LLM-evolution-logs_*.jsonl"),
        os.path.join(LOGS_DIR, "pod_*.log"),
        # Fallback to base dir for older logs
        os.path.join(BASE_DIR, "ga_evaluations_*.jsonl"),
        os.path.join(BASE_DIR, "LLM-evolution-logs_*.jsonl"),
    ]
    
    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=os.path.getmtime)
            match = ts_pattern.search(os.path.basename(latest))
            if match:
                is_cifar100 = "cifar100" in os.path.basename(latest)
                return match.group(1), is_cifar100
    
    # Final fallback: current wall-clock time
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats_records(target_ts=None):
    """
    Read the ga_evaluations*.jsonl to get exact chronological evaluation order for the current run.
    """
    records = []
    if target_ts:
        log_files = [
            os.path.join(BASE_DIR, "logs_cifar10", f"ga_evaluations_{target_ts}.jsonl"),
            os.path.join(BASE_DIR, "logs_cifar100", f"ga_evaluations_{target_ts}.jsonl"),
            os.path.join(LOGS_DIR, f"ga_evaluations_{target_ts}.jsonl"),
            os.path.join(BASE_DIR, f"ga_evaluations_{target_ts}.jsonl")
        ]
        log_files = [f for f in log_files if os.path.exists(f)]
    else:
        log_files = glob.glob(os.path.join(BASE_DIR, "logs_cifar10", "ga_evaluations*.jsonl")) + \
                    glob.glob(os.path.join(BASE_DIR, "logs_cifar100", "ga_evaluations*.jsonl")) + \
                    glob.glob(os.path.join(LOGS_DIR, "ga_evaluations*.jsonl"))
        if not log_files:
            log_files = glob.glob(os.path.join(BASE_DIR, "ga_evaluations*.jsonl"))
            
    if not log_files:
        _warn(f"No ga_evaluations{'* ' if not target_ts else '_'+target_ts}.jsonl files found")
        return records
        
    latest_log = log_files[0] if target_ts else max(log_files, key=os.path.getmtime)
    print(f"  [Info] Loading GA eval logs from: {os.path.basename(latest_log)}")
    
    with open(latest_log) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                records.append(data)
            except Exception as e:
                _warn(f"Could not read line {i+1} in {latest_log}: {e}")

    return records


def split_into_generations(entries, gen1_size=20, rest_size=15):
    generations = []
    if not entries: return generations
    # Generation 1
    generations.append(entries[:gen1_size])
    idx = gen1_size
    # Remaining generations
    while idx < len(entries):
        generations.append(entries[idx:idx + rest_size])
        idx += rest_size
    return generations


def load_llm_logs(target_ts=None):
    """
    Read the LLM-evolution-logs*.jsonl. Returns list of dicts.
    Expected fields: method, score, reward, valid_syntax, timestamp.
    """
    if target_ts:
        log_files = [
            os.path.join(BASE_DIR, "logs_cifar10", f"LLM-evolution-logs_{target_ts}.jsonl"),
            os.path.join(BASE_DIR, "logs_cifar100", f"LLM-evolution-logs_{target_ts}.jsonl"),
            os.path.join(LOGS_DIR, f"LLM-evolution-logs_{target_ts}.jsonl"),
            os.path.join(BASE_DIR, f"LLM-evolution-logs_{target_ts}.jsonl")
        ]
        log_files = [f for f in log_files if os.path.exists(f)]
    else:
        log_files = glob.glob(os.path.join(BASE_DIR, "logs_cifar10", "LLM-evolution-logs*.jsonl")) + \
                    glob.glob(os.path.join(BASE_DIR, "logs_cifar100", "LLM-evolution-logs*.jsonl")) + \
                    glob.glob(os.path.join(LOGS_DIR, "LLM-evolution-logs*.jsonl"))
        if not log_files:
            log_files = glob.glob(os.path.join(BASE_DIR, "LLM-evolution-logs*.jsonl"))
            
    if not log_files:
        _warn(f"No LLM-evolution-logs{'* ' if not target_ts else '_'+target_ts}.jsonl files found")
        return []
        
    latest_log = log_files[0] if target_ts else max(log_files, key=os.path.getmtime)
    print(f"  [Info] Loading LLM logs from: {os.path.basename(latest_log)}")
    
    entries = []
    with open(latest_log) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                _warn(f"Skipping malformed JSONL line {i+1}: {e}")
    return entries


# ---------------------------------------------------------------------------
# GA Evolution plots
# ---------------------------------------------------------------------------

def plot_generation_accuracy(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping generation_accuracy.png")
        return
        
    generations = split_into_generations(records, 20, 15)
    gen_numbers, avg_accuracies, peak_accuracies, running_peaks = [], [], [], []
    running_peak = 0.0
    
    for i, gen in enumerate(generations):
        gen_num = i + 1
        accuracies = [e.get("accuracy", 0) for e in gen]
        if not accuracies: continue
        avg_acc = np.mean(accuracies)
        peak_acc = max(accuracies)
        running_peak = max(running_peak, peak_acc)
        
        gen_numbers.append(gen_num)
        avg_accuracies.append(avg_acc)
        peak_accuracies.append(peak_acc)
        running_peaks.append(running_peak)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(gen_numbers, avg_accuracies, label="Average Accuracy (per gen)",
                color="#3b82f6", linewidth=1.5, alpha=0.8, marker=".", markersize=4)
        ax.plot(gen_numbers, peak_accuracies, label="Peak Accuracy (per gen)",
                color="#f97316", linewidth=1.5, alpha=0.8, marker=".", markersize=4)
        ax.plot(gen_numbers, running_peaks, label="Running Best (cumulative)",
                color="#10b981", linewidth=2.5, linestyle="--")

        ax.set_xlabel("Generation", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title("LLM-Guided GA: Accuracy per Generation (No Fractal Drop Path)", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        
        # Override grid and ticks to exactly match baseline visual appeal
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(colors="black")
        
        if generations:
            ax.set_xlim(1, len(generations))
            ax.set_xticks(np.arange(0, len(generations) + 1, 20))
            
        if running_peaks:
            upper_limit = min(100, max(running_peaks) + 5)
            ax.set_ylim(30, upper_limit)
        else:
            ax.set_ylim(30, 100)

        if running_peaks:
            ax.annotate(f"{running_peaks[-1]:.2f}%",
                        xy=(gen_numbers[-1], running_peaks[-1]),
                        xytext=(-60, 15), textcoords="offset points",
                        fontsize=11, fontweight="bold", color="#10b981",
                        arrowprops=dict(arrowstyle="->", color="#10b981"))
                        
        plt.tight_layout()
        path = os.path.join(out_dir, "generation_accuracy.png")
        _save(fig, path, saved_files)


def plot_population_diversity(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping population_diversity.png")
        return

    accs = [r["accuracy"] * 100 for r in records if r["accuracy"] is not None]
    if not accs:
        _warn("No accuracy values found — skipping population_diversity.png")
        return

    # Group into batches to simulate generation-level diversity
    batch_size = int(os.environ.get("POPULATION_SIZE", 20))
    elites = 5  # Elites are carried forward and not re-evaluated
    batches, labels = [], []
    
    i = 0
    gen_idx = 1
    while i < len(accs):
        current_batch_size = batch_size if gen_idx == 1 else (batch_size - elites)
        chunk = accs[i:i + current_batch_size]
        if chunk:
            batches.append(chunk)
            labels.append(f"{gen_idx}")
        i += current_batch_size
        gen_idx += 1

    with plt.rc_context(PLOT_STYLE):
        # Cap the width at 24 inches so it doesn't become too huge
        fig, ax = plt.subplots(figsize=(min(24, max(6, len(batches) * 0.5)), 5))
        bp = ax.boxplot(
            batches,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="#2a3a6e", color=ACCENT1),
            medianprops=dict(color=ACCENT2, linewidth=2),
            whiskerprops=dict(color="#6a7aad"),
            capprops=dict(color="#6a7aad"),
            flierprops=dict(marker="o", color=ACCENT3, alpha=0.5, markersize=4),
        )
        
        if len(batches) > 20:
            step = max(1, len(batches) // 20)
            ax.set_xticks(list(range(1, len(batches) + 1, step)))
            ax.set_xticklabels([labels[i] for i in range(0, len(batches), step)], rotation=45, ha='right')
        else:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
        _apply_style(ax, "Population Diversity (Accuracy Spread per Generation Batch)",
                     "Generation Batch", "Accuracy (%)")
        _save(fig, os.path.join(out_dir, "population_diversity.png"), saved_files)


def plot_best_vs_avg_accuracy(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping best_vs_avg_accuracy.png")
        return

    accs = [r["accuracy"] for r in records if r["accuracy"] is not None]
    bests = [r.get("best_accuracy") for r in records if r.get("best_accuracy") is not None]
    if not accs:
        _warn("No accuracy values — skipping best_vs_avg_accuracy.png")
        return

    batch_size = int(os.environ.get("POPULATION_SIZE", 20))
    elites = 5
    avg_per_batch, best_per_batch, median_per_batch, ci_per_batch, gen_labels = [], [], [], [], []
    
    i = 0
    gen_idx = 1
    while i < max(len(accs), len(bests)):
        current_batch_size = batch_size if gen_idx == 1 else (batch_size - elites)
        chunk_acc  = accs[i:i + current_batch_size]
        chunk_best = bests[i:i + current_batch_size] if bests else []
        if not chunk_acc:
            break
        avg = sum(chunk_acc) / len(chunk_acc)
        avg_per_batch.append(avg)
        best_per_batch.append(max(chunk_best) if chunk_best else max(chunk_acc))
        median_per_batch.append(np.median(chunk_acc))
        
        # Calculate 95% Confidence Interval for the mean
        std = np.std(chunk_acc, ddof=1) if len(chunk_acc) > 1 else 0
        ci = 1.96 * (std / np.sqrt(len(chunk_acc)))
        ci_per_batch.append(ci)
        gen_labels.append(f"{gen_idx}")
        i += current_batch_size
        gen_idx += 1

    xs = list(range(len(gen_labels)))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(min(24, max(6, len(xs) * 0.5)), 5))
        
        # Use a line plot instead of bars for better readability on long runs
        ax.plot(xs, avg_per_batch, color=BAR_COLOR, alpha=0.8, label="Avg Accuracy", zorder=2, linewidth=2)
        
        # Add the shaded 95% Confidence Interval band
        lower_bound = np.array(avg_per_batch) - np.array(ci_per_batch)
        upper_bound = np.array(avg_per_batch) + np.array(ci_per_batch)
        ax.fill_between(xs, lower_bound, upper_bound, alpha=0.2, color=BAR_COLOR, zorder=1, label="95% CI (Avg)")
        
        ax.plot(xs, median_per_batch, color="#2ca02c", alpha=0.9, linestyle="--", label="Median Accuracy", zorder=2, linewidth=2)
        
        ax.plot(xs, best_per_batch, color=ACCENT1, linewidth=2.5, marker="D",
                markersize=4, label="Best Accuracy", zorder=3)
                
        if len(xs) > 20:
            step = max(1, len(xs) // 20)
            ax.set_xticks(xs[::step])
            ax.set_xticklabels([gen_labels[i] for i in range(0, len(xs), step)], rotation=45, ha='right')
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(gen_labels, rotation=45, ha='right')
            
        _apply_style(ax, "Best vs Average Accuracy per Generation Batch",
                     "Generation Batch", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "best_vs_avg_accuracy.png"), saved_files)


def plot_time_per_generation(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping time_per_generation.png")
        return

    generations = split_into_generations(records, 20, 15)
    gen_numbers = []
    gen_times = []
    prev_end_time = None
    
    for i, gen in enumerate(generations):
        gen_num = i + 1
        gen_numbers.append(gen_num)
        
        times = [datetime.fromisoformat(e["timestamp"]) for e in gen if "timestamp" in e]
        if times:
            gen_start_time = prev_end_time if prev_end_time else times[0]
            gen_end_time = times[-1]
            gen_duration = (gen_end_time - gen_start_time).total_seconds() / 60.0 # in minutes
            
            if prev_end_time is None and len(times) > 1:
                avg_model_time = (times[-1] - times[0]).total_seconds() / (len(times) - 1)
                gen_duration += avg_model_time / 60.0
                
            gen_times.append(gen_duration)
            prev_end_time = gen_end_time
        else:
            gen_times.append(0.0)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(gen_numbers, gen_times, label="Time Taken (per gen)",
                color="#a855f7", linewidth=2.0, alpha=0.9, marker="s", markersize=5)
        
        ax.set_xlabel("Generation", fontsize=13)
        ax.set_ylabel("Time Taken (Minutes)", fontsize=13)
        ax.set_title("LLM-Guided GA: Compute Time per Generation", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right")
        
        # Override grid and ticks to exactly match baseline visual appeal
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(colors="black")
        
        if generations:
            ax.set_xlim(1, len(generations))
            
        plt.tight_layout()
        path = os.path.join(out_dir, "time_per_generation.png")
        _save(fig, path, saved_files)


# ---------------------------------------------------------------------------
# LLM fine-tuning plots
# ---------------------------------------------------------------------------

def plot_reward_over_iterations(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping reward_over_iterations.png")
        return

    rewards = [e.get("reward", 0.0) for e in entries]
    xs = list(range(1, len(rewards) + 1))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [ACCENT3 if r > 0 else ACCENT2 for r in rewards]
        ax.bar(xs, rewards, color=colors, alpha=0.85, zorder=2)
        ax.axhline(0, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.5, label="y = 0")
        ax.plot(xs, rewards, color=ACCENT1, linewidth=1.5, alpha=0.7)
        _apply_style(ax, "RL Reward Over Meta-Evolution Iterations",
                     "Iteration", "Reward")
        pos_patch = mpatches.Patch(color=ACCENT3, label="Positive reward")
        neg_patch = mpatches.Patch(color=ACCENT2, label="Penalty")
        ax.legend(handles=[pos_patch, neg_patch])
        _save(fig, os.path.join(out_dir, "reward_over_iterations.png"), saved_files)


def plot_syntax_success_rate(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping syntax_success_rate.png")
        return

    valid = [1 if e.get("valid_syntax", False) else 0 for e in entries]
    xs = list(range(1, len(valid) + 1))
    window = 10

    # Rolling success rate
    rolling = []
    for i in range(len(valid)):
        start = max(0, i - window + 1)
        chunk = valid[start:i + 1]
        rolling.append(sum(chunk) / len(chunk) * 100)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(xs, rolling, alpha=0.2, color=ACCENT1)
        ax.plot(xs, rolling, color=ACCENT1, linewidth=2, label=f"Rolling success rate (window={window})")
        ax.axhline(50, color=ACCENT2, linestyle="--", linewidth=1, alpha=0.6, label="50% baseline")
        ax.set_ylim(0, 105)
        _apply_style(ax, "Syntax Success Rate Over Iterations",
                     "Iteration", "Success Rate (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "syntax_success_rate.png"), saved_files)


def plot_score_improvement(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping score_improvement.png")
        return

    # Only use entries that have both score fields
    filtered = [e for e in entries if "score" in e]
    if not filtered:
        _warn("No 'score' fields in LLM logs — skipping score_improvement.png")
        return

    xs      = list(range(1, len(filtered) + 1))
    scores  = [e.get("score", 0.0)         for e in filtered]
    rewards = [e.get("reward", 0.0)         for e in filtered]
    # Derive baseline: baseline = score - reward (since reward = score - baseline in meta_evolver)
    baselines = [max(0.0, s - r) for s, r in zip(scores, rewards)]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, baselines, color=ACCENT2, linewidth=2, linestyle="--",
                marker="o", markersize=4, label="Baseline Score")
        ax.plot(xs, scores,   color=ACCENT3, linewidth=2,
                marker="s", markersize=4, label="New Score")
        ax.fill_between(xs, baselines, scores,
                        where=[s > b for s, b in zip(scores, baselines)],
                        alpha=0.2, color=ACCENT3, label="Improvement region")
        ax.fill_between(xs, baselines, scores,
                        where=[s <= b for s, b in zip(scores, baselines)],
                        alpha=0.15, color=ACCENT2, label="Regression region")
        _apply_style(ax, "Score Improvement per Iteration (LLM Fine-Tuning)",
                     "Iteration", "Score")
        ax.legend()
        _save(fig, os.path.join(out_dir, "score_improvement.png"), saved_files)


def plot_peak_accuracy_over_iterations(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping meta_peak_accuracy.png")
        return
    accs = [e.get("peak_accuracy", 0.0) for e in entries]
    xs = list(range(1, len(accs) + 1))
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, accs, color=ACCENT1, linewidth=2.5, marker="o", markersize=5, label="Peak Accuracy")
        ax.fill_between(xs, accs, alpha=0.15, color=ACCENT1)
        _apply_style(ax, "Peak GA Accuracy Over Meta-Iterations", "Meta-Iteration", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "meta_peak_accuracy.png"), saved_files)


def plot_modification_success_rate(entries, out_dir, saved_files):
    if not entries:
        _warn("No LLM log entries — skipping llm_success_rates.png")
        return
    syntax_valid = [1 if e.get("valid_syntax", False) else 0 for e in entries]
    improved = [1 if e.get("reward", 0.0) > 0 else 0 for e in entries]
    xs = list(range(1, len(syntax_valid) + 1))
    window = min(10, max(2, len(entries) // 5))  # Dynamic window, capped
    roll_syntax = [sum(syntax_valid[max(0, i-window+1):i+1]) / len(syntax_valid[max(0, i-window+1):i+1]) * 100 for i in range(len(syntax_valid))]
    roll_improve = [sum(improved[max(0, i-window+1):i+1]) / len(improved[max(0, i-window+1):i+1]) * 100 for i in range(len(improved))]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, roll_syntax, color=ACCENT1, linewidth=2, label=f"Syntax Success (rolling)")
        ax.plot(xs, roll_improve, color=ACCENT3, linewidth=2, linestyle="--", label=f"Improvement Success (rolling)")
        ax.axhline(50, color="#ffffff", linestyle=":", linewidth=1, alpha=0.3)
        ax.set_ylim(-5, 105)
        _apply_style(ax, "LLM Modification Success Rates", "Meta-Iteration", "Success Rate (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "llm_success_rates.png"), saved_files)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    target_ts = None
    if len(sys.argv) > 1:
        target_ts = sys.argv[1]
        
    # Use source log timestamp so visualizations correlate with their experiment
    timestamp, is_cifar100 = _extract_log_timestamp(target_ts)
    prefix = "cifar100_" if is_cifar100 else ""
    run_dir    = os.path.join(VIZ_ROOT, f"run_{prefix}{timestamp}")
    ga_dir     = os.path.join(run_dir, "ga_evolution")
    ft_dir     = os.path.join(run_dir, "fine_tuning")

    os.makedirs(ga_dir, exist_ok=True)
    os.makedirs(ft_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  visualize_meta_generation.py — run: {timestamp}")
    print(f"  Output root: {os.path.relpath(run_dir, BASE_DIR)}")
    print(f"{'='*60}\n")

    saved_files = []

    # ── GA evolution ────────────────────────────────────────────────────────
    print("[1/2] Loading stats records …")
    records = load_stats_records(target_ts)
    print(f"      Found {len(records)} evaluated model(s).\n")

    print("  Generating GA evolution plots …")
    plot_generation_accuracy(records,     ga_dir, saved_files)
    plot_time_per_generation(records,     ga_dir, saved_files)
    plot_population_diversity(records,    ga_dir, saved_files)
    plot_best_vs_avg_accuracy(records,    ga_dir, saved_files)

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    print("\n[2/2] Loading LLM evolution logs …")
    entries = load_llm_logs(target_ts)
    print(f"      Found {len(entries)} log entry(ies).\n")

    print("  Generating fine-tuning plots …")
    plot_reward_over_iterations(entries, ft_dir, saved_files)
    # plot_syntax_success_rate(entries,   ft_dir, saved_files)  # Replaced by plot_modification_success_rate
    plot_score_improvement(entries,     ft_dir, saved_files)
    plot_peak_accuracy_over_iterations(entries, ft_dir, saved_files)
    plot_modification_success_rate(entries, ft_dir, saved_files)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Saved {len(saved_files)} plot(s):")
    for p in saved_files:
        print(f"    • {os.path.relpath(p, BASE_DIR)}")
    if not saved_files:
        print("    (none — check warnings above)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
