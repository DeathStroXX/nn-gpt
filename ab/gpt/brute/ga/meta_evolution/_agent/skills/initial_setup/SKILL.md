---
name: initial_setup
description: Guide for initializing the meta-evolution environment and resolving common pipeline import/loader bugs.
---

# Meta-Evolution Initial Setup & Debugging Guide

This skill documents the necessary environment configuration and critical bug fixes required to run the Meta-Evolution and LLM-guided pipelines effectively.

## 1. Environment & Venv Setup

### Workflow
1. **Local Venv**:
   - Location: `/home/b-a-singh/Thesis/nn-gpt/.venv`
   - Activation: `source /home/b-a-singh/Thesis/nn-gpt/.venv/bin/activate`
   - **Note**: Avoid `pip install -r requirements.txt` if you want to maintain compatibility with the pre-installed CUDA/Torch drivers in specific environments. Use `pip install git+https://github.com/ABrain-One/nn-dataset` instead.

2. **Kubernetes Environment**:
   - Image: `abrainone/ai-linux:cu12.6.3-latest`
   - Workspace: `/a/mm` (mounted from `/shared/ssd/home/b-a-singh/Thesis/nn-gpt`)
   - PYTHONPATH: Must include `/a/mm` to locate the `ab` package.

### Critical Dependency Installation
To ensure the pipeline has all necessary neural network utilities, always upgrade the `nn-dataset` package:
```bash
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```

## 2. Documented Bug Fixes

### A. The "no such table: loader" Evaluation Error
**Problem:** Evaluation fails with `Evaluation error (Eval): no such table: loader`.
**Cause:** The `nn-dataset` library fails to find the physical `ab/nn/loader` module files because `PYTHONPATH` points to a local volume that doesn't contain the installed library sub-packages. It then defaults to a legacy SQLite lookup for a table that does not exist.
**Solution:** Force-copy the installed site-packages into the local workspace before running the script:
```bash
site_pkg=$(python3 -c 'import site; print(site.getusersitepackages())')
mkdir -p /a/mm/ab/nn
cp -rn $site_pkg/ab/nn/* /a/mm/ab/nn/ || true
```

### B. The `flash_attn` / `torch` ABI Mismatch
**Problem:** `ImportError: ... undefined symbol: _ZNK3c106SymInt6sym_neERKS0_` when importing `transformers` or `peft`.
**Cause:** Installing `torch==2.9.1` (often via `requirements.txt`) breaks compatibility with the `flash-attn` version pre-compiled in the `ai-linux` image.
**Solution:**
1. **Prefer Image Torch**: Do not use `pip install -r requirements.txt` if it pins a specific torch version that overrides the image's native torch.
2. **Uninstall Conflict**: If errors persist, uninstall the conflicting `flash-attn` package:
   ```bash
   pip uninstall -y flash-attn
   ```
   Or ensure the installed torch version matches the one the image was built with.

## 3. Running Jobs
Always use the updated `args` in `tune_meta.json` or `meta_evol_tune_nngpt.json` that incorporates these filesystem fixes.
