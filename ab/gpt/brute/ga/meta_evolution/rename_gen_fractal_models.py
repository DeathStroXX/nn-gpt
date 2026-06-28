#!/usr/bin/env python3
"""
Migration script: Rename GA fractal model artifacts from FractalNet-{checksum}
to GenFractalNet-{checksum}.

Targets:
  - Model files in ga_fractal_arch/   (FractalNet-{checksum}.py -> GenFractalNet-{checksum}.py)
  - Stats folders in stats/           (img-classification_cifar-10_FractalNet-{checksum} -> ...GenFractalNet-{checksum})
  - Stats folders in best_fractal_stats/ (same pattern)

Usage:
  python rename_gen_fractal_models.py --dry-run   # preview changes
  python rename_gen_fractal_models.py              # apply changes
"""

import os
import re
import argparse

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(BASE_DIR, "ga_fractal_arch")
STATS_DIR = os.path.join(BASE_DIR, "stats")
BEST_STATS_DIR = os.path.join(BASE_DIR, "best_fractal_stats")

# Patterns
MODEL_OLD_PREFIX = "FractalNet-"
MODEL_NEW_PREFIX = "GenFractalNet-"
STATS_OLD_PREFIX = "img-classification_cifar-10_FractalNet-"
STATS_NEW_PREFIX = "img-classification_cifar-10_acc_GenFractalNet-"

# Regex to extract checksum from old-named model files (e.g. FractalNet-abc123.py)
MODEL_RE = re.compile(r"^FractalNet-([a-f0-9]+)(\.py)?$")
# Regex to extract checksum from old-named stats folders
STATS_RE = re.compile(r"^img-classification_cifar-10_FractalNet-([a-f0-9]+)$")


def _rename_models(dry_run: bool):
    """Rename model files in ga_fractal_arch/."""
    renamed = 0
    skipped_already_new = 0
    skipped_conflict = 0
    errors = []

    if not os.path.isdir(ARCH_DIR):
        print(f"[models] Directory not found: {ARCH_DIR}")
        return renamed, skipped_already_new, skipped_conflict, errors

    for name in sorted(os.listdir(ARCH_DIR)):
        # Skip items already using new naming
        if name.startswith(MODEL_NEW_PREFIX):
            skipped_already_new += 1
            continue

        match = MODEL_RE.match(name)
        if not match:
            continue  # not a FractalNet model file, ignore

        checksum = match.group(1)
        ext = match.group(2) or ""
        new_name = f"{MODEL_NEW_PREFIX}{checksum}{ext}"

        src = os.path.join(ARCH_DIR, name)
        dst = os.path.join(ARCH_DIR, new_name)

        if os.path.exists(dst):
            msg = f"  [CONFLICT] {name} -> {new_name}  (target already exists, skipping)"
            print(msg)
            skipped_conflict += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] {name} -> {new_name}")
        else:
            try:
                os.rename(src, dst)
                print(f"  [RENAMED] {name} -> {new_name}")
            except OSError as e:
                msg = f"  [ERROR]   {name} -> {new_name}: {e}"
                print(msg)
                errors.append(msg)
                continue

        renamed += 1

    return renamed, skipped_already_new, skipped_conflict, errors


def _rename_stats_folders(stats_dir: str, label: str, dry_run: bool):
    """Rename stats folders in the given directory."""
    renamed = 0
    skipped_already_new = 0
    skipped_conflict = 0
    errors = []

    if not os.path.isdir(stats_dir):
        print(f"[{label}] Directory not found: {stats_dir}")
        return renamed, skipped_already_new, skipped_conflict, errors

    for name in sorted(os.listdir(stats_dir)):
        # Skip items already using new naming
        if name.startswith(STATS_NEW_PREFIX):
            skipped_already_new += 1
            continue

        match = STATS_RE.match(name)
        if not match:
            continue  # not a matching stats folder, ignore

        checksum = match.group(1)
        new_name = f"{STATS_NEW_PREFIX}{checksum}"

        src = os.path.join(stats_dir, name)
        dst = os.path.join(stats_dir, new_name)

        if not os.path.isdir(src):
            continue  # only rename directories

        if os.path.exists(dst):
            msg = f"  [CONFLICT] {name} -> {new_name}  (target already exists, skipping)"
            print(msg)
            skipped_conflict += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] {name} -> {new_name}")
        else:
            try:
                os.rename(src, dst)
                print(f"  [RENAMED] {name} -> {new_name}")
            except OSError as e:
                msg = f"  [ERROR]   {name} -> {new_name}: {e}"
                print(msg)
                errors.append(msg)
                continue

        renamed += 1

    return renamed, skipped_already_new, skipped_conflict, errors


def main():
    parser = argparse.ArgumentParser(
        description="Rename FractalNet-{checksum} artifacts to GenFractalNet-{checksum}."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned renames without making any changes.",
    )
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"\n{'='*60}")
    print(f"  FractalNet -> GenFractalNet Migration  [{mode}]")
    print(f"{'='*60}\n")

    # --- 1. Model files ---
    print(f"--- Model files ({ARCH_DIR}) ---")
    m_renamed, m_skip_new, m_skip_conflict, m_errors = _rename_models(args.dry_run)

    # --- 2. Stats folders ---
    print(f"\n--- Stats folders ({STATS_DIR}) ---")
    s_renamed, s_skip_new, s_skip_conflict, s_errors = _rename_stats_folders(
        STATS_DIR, "stats", args.dry_run
    )

    # --- 3. Best stats folders ---
    print(f"\n--- Best-stats folders ({BEST_STATS_DIR}) ---")
    b_renamed, b_skip_new, b_skip_conflict, b_errors = _rename_stats_folders(
        BEST_STATS_DIR, "best_stats", args.dry_run
    )

    # --- Summary ---
    total_renamed = m_renamed + s_renamed + b_renamed
    total_skip_new = m_skip_new + s_skip_new + b_skip_new
    total_conflict = m_skip_conflict + s_skip_conflict + b_skip_conflict
    total_errors = len(m_errors) + len(s_errors) + len(b_errors)

    action_word = "to rename" if args.dry_run else "renamed"

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Models {action_word}:          {m_renamed}")
    print(f"  Stats folders {action_word}:   {s_renamed}")
    print(f"  Best-stats {action_word}:      {b_renamed}")
    print(f"  ─────────────────────────────────")
    print(f"  Total {action_word}:           {total_renamed}")
    print(f"  Already GenFractalNet (skip): {total_skip_new}")
    print(f"  Conflicts (skip):            {total_conflict}")
    print(f"  Errors:                      {total_errors}")
    print(f"{'='*60}")

    if args.dry_run and total_renamed > 0:
        print(f"\n  Re-run without --dry-run to apply these renames.\n")
    elif not args.dry_run and total_renamed > 0:
        print(f"\n  Migration complete.\n")
    else:
        print(f"\n  Nothing to rename.\n")


if __name__ == "__main__":
    main()
