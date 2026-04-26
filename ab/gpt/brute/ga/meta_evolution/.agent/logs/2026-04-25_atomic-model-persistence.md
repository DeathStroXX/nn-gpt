# Atomic Model Persistence — Write-Model-Last Safeguard

**Date:** 2026-04-25
**File changed:** `run_fractal_evolution.py`
**Scope:** `fitness_function()` only

---

## Problem

Model architecture files were written to `ga_fractal_arch/` **before** evaluation
started (line 141, original). If evaluation crashed, was interrupted, or stats
writing failed, the model file remained on disk — an "orphan" with no
corresponding `stats/` directory. This made it impossible to distinguish
evaluated-vs-unevaluated architectures and wasted disk space.

## Solution: Write-Model-Last

The model `.py` file is now written to a **temporary path** (`_tmp_GenFractalNet-{checksum}.py`)
during evaluation. It is only promoted to the final path (`GenFractalNet-{checksum}.py`)
after:

1. Evaluation completes without exception.
2. Stats JSON files are written to `stats/img-classification_cifar-10_GenFractalNet-{checksum}/`.
3. At least one `.json` file is **verified to exist on disk** in that stats directory.

## Changes Made

### 1. Sentinel variables (new, lines 125-127)
```python
tmp_filepath = None
model_stats_dir_path = None
```
Initialized at the top of the `try` block so the `except` cleanup can safely
reference them even if the exception fires before the variables would normally
be assigned.

### 2. Temporary file write (lines 141-152, replaces old lines 137-142)
```python
# Old (commented out):
# filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
# with open(filepath, 'w') as f:
#     f.write(code_str)

# New:
final_filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
tmp_filepath   = os.path.join(ARCH_DIR, f"_tmp_{model_name}.py")
filepath       = tmp_filepath  # evaluator works on the temp file
with open(tmp_filepath, 'w') as f:
    f.write(code_str)
```

### 3. Stats verification + promote (new, lines 269-282)
After stats JSON files are written, we verify at least one `.json` exists:
```python
_stats_json_files = [f for f in os.listdir(model_stats_dir_path) if f.endswith('.json')]
if not _stats_json_files:
    # Clean up temp file + empty stats dir, return 0.0
    ...
os.rename(tmp_filepath, final_filepath)
```

### 4. Enhanced except block with cleanup (lines 329-343)
```python
except Exception as e:
    ...
    # Cleanup: remove temp model file and any partial stats
    if tmp_filepath and os.path.exists(tmp_filepath):
        os.remove(tmp_filepath)
    if model_stats_dir_path and os.path.isdir(model_stats_dir_path):
        shutil.rmtree(model_stats_dir_path)
```

## New Save Order

```
1. Generate code string in memory
2. Compute checksum (dedup check)
3. Write code to _tmp_GenFractalNet-{checksum}.py  (temporary)
4. Run Eval engine (training + evaluation)
5. Collect stats from training_summary.json / result object
6. Write stats JSON(s) to stats/img-classification_cifar-10_GenFractalNet-{checksum}/
7. VERIFY at least one .json exists in that stats directory
8. os.rename(_tmp_... → GenFractalNet-{checksum}.py)  (atomic on same filesystem)
9. Extract accuracy, add to seen_checksums, return fitness
```

## Failure Handling

| Failure Point | What happens |
|---|---|
| Exception during eval (step 4) | `except` block removes `_tmp_*.py` + partial `stats/` dir |
| Stats write fails (step 6) | `except` block catches the I/O error, same cleanup |
| Stats dir exists but empty (step 7) | Explicit verification removes temp file + empty stats dir, returns 0.0 |
| Interrupt (SIGINT/SIGKILL) | Temp files use `_tmp_` prefix — easy to identify and clean manually |

## Naming Consistency Audit

All naming paths now use `GenFractalNet-{checksum}` consistently:

| Path | Pattern |
|---|---|
| Model file save | `GenFractalNet-{checksum}.py` (line 143) |
| Stats dir write | `img-classification_cifar-10_GenFractalNet-{checksum}` (line 240) |
| Duplicate detection / checksum load | `img-classification_cifar-10_GenFractalNet-` prefix (line 58) |
| Stats lookup for duplicates | `img-classification_cifar-10_GenFractalNet-{checksum}` (line 76) |
| Best model stats copy | `img-classification_cifar-10_GenFractalNet-{checksum}` (line 376) |

No `FractalNet` vs `GenFractalNet` mismatch exists.

## Existing Orphan Models

Since `ga_fractal_arch/` and `stats/` don't exist yet in the `clone3` workspace
(they are created at runtime), there are **no existing orphan models** to clean up.

For the original `nn-gpt` workspace, a one-liner to find orphans:
```bash
# List model files in ga_fractal_arch/ with no matching stats/ directory
for f in ga_fractal_arch/GenFractalNet-*.py; do
  cs=$(echo "$f" | grep -oP '[a-f0-9]{32}')
  [ ! -d "stats/img-classification_cifar-10_GenFractalNet-$cs" ] && echo "ORPHAN: $f"
done
```

## Verification Trace

### Path A — Successful evaluation
1. `generate_model_code_string()` → code string
2. `uuid4()` → checksum `abc123...`
3. Write to `ga_fractal_arch/_tmp_GenFractalNet-abc123....py`
4. `Eval.evaluate()` completes, returns result
5. Stats written to `stats/img-classification_cifar-10_GenFractalNet-abc123.../1.json`
6. `os.listdir()` confirms `['1.json']` → non-empty ✓
7. `os.rename(_tmp_... → GenFractalNet-abc123....py)` ✓
8. Return fitness score

### Path B — Evaluation crash
1. Steps 1-3 same as above
4. `Eval.evaluate()` raises `RuntimeError`
5. `except` block fires:
   - `tmp_filepath` is set → `os.remove(_tmp_*.py)` ✓
   - `model_stats_dir_path` is `None` (never assigned) → skip ✓
6. Return 0.0, no orphan files left

### Path C — Stats directory created but write fails
1. Steps 1-5 same as Path A
6. `os.makedirs(model_stats_dir_path)` succeeds
7. `json.dump()` raises `IOError`
8. `except` block fires:
   - `os.remove(_tmp_*.py)` ✓
   - `shutil.rmtree(model_stats_dir_path)` removes empty dir ✓
9. Return 0.0, no orphan files left
