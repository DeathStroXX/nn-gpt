"""
visualize_training.py
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
    python3 visualize_training.py
"""

import os
import json
import warnings
import glob
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3f5c",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#b0b0b0",
    "ytick.color":      "#b0b0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "legend.facecolor": "#1a1d2e",
    "legend.edgecolor": "#3a3f5c",
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
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(colors="#b0b0b0")


def _save(fig, path, saved_files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(path)
    print(f"  [saved] {os.path.relpath(path, BASE_DIR)}")


def _warn(msg):
    print(f"  [WARN]  {msg}")

def _extract_log_timestamp():
    """
    Extract the experiment timestamp from source log filenames.
    Priority: ga_evaluations > LLM-evolution-logs > pod log > fallback to now().
    """
    ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')
    
    # Priority order of log file patterns to check
    search_patterns = [
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
                return match.group(1)
    
    # Final fallback: current wall-clock time
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats_records():
    """
    Read the latest ga_evaluations*.jsonl to get exact chronological evaluation order for the current run.
    """
    records = []
    log_files = glob.glob(os.path.join(LOGS_DIR, "ga_evaluations*.jsonl"))
    if not log_files:
        # Fallback to base dir for older logs
        log_files = glob.glob(os.path.join(BASE_DIR, "ga_evaluations*.jsonl"))
    if not log_files:
        _warn("No ga_evaluations*.jsonl files found")
        return records
        
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"  [Info] Loading GA eval logs from: {os.path.basename(latest_log)}")
    
    with open(latest_log) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                # the log saves percentage (e.g. 65.4), but old code multiplied by 100
                # so we divide by 100 here to be compatible with other plot functions
                acc = data.get("accuracy", 0) / 100.0 
                records.append({
                    "uid": data.get("uid"),
                    "accuracy": acc,
                    "best_accuracy": acc,  # For compatibility with best_vs_avg plot
                    "is_cached": data.get("is_cached", False)
                })
            except Exception as e:
                _warn(f"Could not read line {i+1} in {latest_log}: {e}")

    return records


def load_llm_logs():
    """
    Read the latest LLM-evolution-logs*.jsonl. Returns list of dicts.
    Expected fields: method, score, reward, valid_syntax, timestamp.
    """
    log_files = glob.glob(os.path.join(LOGS_DIR, "LLM-evolution-logs*.jsonl"))
    if not log_files:
        # Fallback to base dir for older logs
        log_files = glob.glob(os.path.join(BASE_DIR, "LLM-evolution-logs*.jsonl"))
    if not log_files:
        _warn("No LLM-evolution-logs*.jsonl files found")
        return []
        
    latest_log = max(log_files, key=os.path.getmtime)
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

    x = list(range(1, len(records) + 1))
    acc = [r["accuracy"] * 100 if r["accuracy"] is not None else None for r in records]
    is_cached = [r.get("is_cached", False) for r in records]

    # Split points into new vs cached
    x_new, y_new = [], []
    x_cached, y_cached = [], []
    
    for xi, yi, cached in zip(x, acc, is_cached):
        if yi is not None:
            if cached:
                x_cached.append(xi)
                y_cached.append(yi)
            else:
                x_new.append(xi)
                y_new.append(yi)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Draw a faint continuous line connecting all evaluations chronologically
        ax.plot(x, acc, color="#ffffff", alpha=0.2, linewidth=1, zorder=1)
        
        # Scatter plot for newly evaluated models
        if x_new:
            ax.scatter(x_new, y_new, color=ACCENT1, s=40, label="New Evaluation", zorder=3, edgecolors="white", linewidths=0.5)
            
        # Scatter plot for cached duplicates
        if x_cached:
            ax.scatter(x_cached, y_cached, color="#5a6a7a", s=30, label="Cached Duplicate", zorder=2, alpha=0.6)
            
        _apply_style(ax, "Model Accuracy Over Evaluation Order", "Evaluation Index", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "generation_accuracy.png"), saved_files)


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
    batches, labels = [], []
    for i in range(0, len(accs), batch_size):
        chunk = accs[i:i + batch_size]
        if chunk:
            batches.append(chunk)
            labels.append(f"Gen {i // batch_size + 1}")

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(batches) * 1.2), 5))
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
        _apply_style(ax, "Population Diversity (Accuracy Spread per Generation Batch)",
                     "Generation Batch", "Accuracy (%)")
        _save(fig, os.path.join(out_dir, "population_diversity.png"), saved_files)


def plot_best_vs_avg_accuracy(records, out_dir, saved_files):
    if not records:
        _warn("No stats records found — skipping best_vs_avg_accuracy.png")
        return

    accs = [r["accuracy"] * 100 for r in records if r["accuracy"] is not None]
    bests = [r["best_accuracy"] * 100 for r in records if r["best_accuracy"] is not None]
    if not accs:
        _warn("No accuracy values — skipping best_vs_avg_accuracy.png")
        return

    batch_size = int(os.environ.get("POPULATION_SIZE", 20))
    avg_per_batch, best_per_batch, gen_labels = [], [], []
    for i in range(0, max(len(accs), len(bests)), batch_size):
        chunk_acc  = accs[i:i + batch_size]
        chunk_best = bests[i:i + batch_size] if bests else []
        if not chunk_acc:
            break
        avg_per_batch.append(sum(chunk_acc) / len(chunk_acc))
        best_per_batch.append(max(chunk_best) if chunk_best else max(chunk_acc))
        gen_labels.append(f"Gen {i // batch_size + 1}")

    xs = list(range(len(gen_labels)))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.2), 5))
        bars = ax.bar(xs, avg_per_batch, color=BAR_COLOR, alpha=0.8, label="Avg Accuracy", zorder=2)
        ax.plot(xs, best_per_batch, color=ACCENT1, linewidth=2.5, marker="D",
                markersize=6, label="Best Accuracy", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(gen_labels)
        _apply_style(ax, "Best vs Average Accuracy per Generation Batch",
                     "Generation Batch", "Accuracy (%)")
        ax.legend()
        _save(fig, os.path.join(out_dir, "best_vs_avg_accuracy.png"), saved_files)


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
    # timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use source log timestamp so visualizations correlate with their experiment
    timestamp  = _extract_log_timestamp()
    run_dir    = os.path.join(VIZ_ROOT, f"run_{timestamp}")
    ga_dir     = os.path.join(run_dir, "ga_evolution")
    ft_dir     = os.path.join(run_dir, "fine_tuning")

    os.makedirs(ga_dir, exist_ok=True)
    os.makedirs(ft_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  visualize_training.py — run: {timestamp}")
    print(f"  Output root: {os.path.relpath(run_dir, BASE_DIR)}")
    print(f"{'='*60}\n")

    saved_files = []

    # ── GA evolution ────────────────────────────────────────────────────────
    print("[1/2] Loading stats records …")
    records = load_stats_records()
    print(f"      Found {len(records)} evaluated model(s).\n")

    print("  Generating GA evolution plots …")
    plot_generation_accuracy(records,     ga_dir, saved_files)
    plot_population_diversity(records,    ga_dir, saved_files)
    plot_best_vs_avg_accuracy(records,    ga_dir, saved_files)

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    print("\n[2/2] Loading LLM evolution logs …")
    entries = load_llm_logs()
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
