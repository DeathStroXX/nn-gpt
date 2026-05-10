# Refactor Log — "Single Stream" Code Generation for FractalNet_evolvable.py

> **Date:** 2026-05-06  
> **File Modified:** `FractalNet_evolvable.py`  
> **Action:** Complete overwrite  
> **Requested By:** User (professor directive for "single stream" output)

---

## Summary of Change

The entire `FractalNet_evolvable.py` was rewritten to eliminate **all conditional logic (`if/elif/else`)** from inside the generated PyTorch model template string. The generated `.py` files now contain only clean, executable code with no dead branches.

---

## What Changed

### Before (Old Paradigm)
- The `generate_model_code_string()` function embedded raw Python `if/else` conditional logic **inside** the `textwrap.dedent(f"""...""")` block.
- Generated model files contained dead branches like:
  ```python
  if "ReLU" == "GELU":
      activation = nn.GELU()
  else:
      activation = nn.ReLU(inplace=True)
  ```
- This produced unnecessarily complex, hard-to-audit output code.

### After (New "Single Stream" Paradigm)
- All gene-to-PyTorch-syntax mapping is now done **before** the template string, using pre-calculated string variables:
  - `act_layer_str` → e.g., `"nn.SiLU(inplace=True)"`
  - `conv_layer_str` → Standard conv or Depthwise separable conv block
  - `norm_layer_str` → `"nn.BatchNorm2d(channels)"` or `"nn.InstanceNorm2d(channels, affine=True)"`
  - `pool_str` → `"nn.MaxPool2d(2)"` or `"nn.AvgPool2d(2)"`
  - `opt_str` → Full optimizer construction string (SGD/AdamW/RMSprop)
- These variables are injected directly into the f-string template, producing a perfectly clean, single-stream output file.

---

## Search Space Expansion

The following **new genes** were added to `SEARCH_SPACE`:

| Gene | Values | Purpose |
|------|--------|---------|
| `conv_type` | `['Standard', 'Depthwise']` | Standard conv vs Depthwise separable conv |
| `norm_type` | `['BatchNorm', 'InstanceNorm']` | Normalization strategy |
| `optimizer_type` | `['SGD', 'AdamW', 'RMSprop']` | Optimizer selection |
| `activation` | `['ReLU', 'GELU', 'LeakyReLU', 'SiLU']` | Expanded activation functions |
| `fc_dropout` | `[0.0, 0.2, 0.5]` | Dropout before final classifier head |

### Existing Genes (Unchanged)
| Gene | Values |
|------|--------|
| `n_columns` | `[2, 3]` |
| `base_channels` | `[16, 32, 64]` |
| `dropout_prob` | `[0.0, 0.1, 0.2, 0.3]` |
| `lr` | `[0.01, 0.005, 0.003, 0.002, 0.001]` |
| `momentum` | `[0.75, 0.8, 0.85, 0.9, 0.92, 0.95]` |
| `n_blocks` | `[2, 3]` |
| `kernel_size` | `[3, 5]` |
| `pooling_type` | `['Max', 'Avg']` |

---

## Critical Design Detail: HASH IDENTIFIERS Block

The generated model code includes a `# --- HASH IDENTIFIERS ---` comment block:

```python
# --- HASH IDENTIFIERS (Ensures unique UUIDs for caching) ---
# LR: 0.01
# Momentum: 0.9
# Activation: SiLU
# Kernel: 3
# Pooling: Max
# Conv Type: Depthwise
# Norm Type: InstanceNorm
# Optimizer: AdamW
# FC Dropout: 0.2
```

**Purpose:** The GA evaluator computes an MD5 checksum of the entire generated code string. Without these identifiers, two architectures that differ only in hyperparameters (e.g., `lr=0.01` vs `lr=0.005`) would produce identical code bodies and collide on the same UUID. This block ensures every unique chromosome maps to a unique checksum.

> **RULE:** Never remove or restructure the HASH IDENTIFIERS block. Doing so will cause UUID collisions, incorrect fitness caching, and corrupted evolution results.

---

## Verification Checklist

- [x] `FractalNet_evolvable.py` completely replaced with new code
- [x] **No `if/elif/else` inside the `f"""..."""` template block** — all conditionals occur beforehand
- [x] `# --- HASH IDENTIFIERS ---` block present in generated string
- [x] All 5 new genes (`conv_type`, `norm_type`, `optimizer_type`, expanded `activation`, `fc_dropout`) present in `SEARCH_SPACE`
- [x] `create_random_chromosome()` and `generate_model_code_string()` exported for GA consumption
