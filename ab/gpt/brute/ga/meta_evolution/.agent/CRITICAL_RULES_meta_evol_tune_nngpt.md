# CRITICAL RULES & ANALYSIS — `meta_evol_tune_nngpt.json`

> **Created:** 2026-05-06  
> **Purpose:** Document the importance, structure, and immutable sections of the Kubernetes Job configuration that powers the entire meta-evolution pipeline.

---

## ⚠️ RULE #1: DO NOT MODIFY LINE 22 (`args`) WITHOUT EXPLICIT USER PERMISSION

**File:** `meta_evol_tune_nngpt.json`  
**Line:** 22  
**Content:**
```
"cd /a/mm && pip uninstall -y flash-attn && pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu126 && export PYTHONPATH=/a/mm && site_pkg=$(python3 -c 'import site; print(site.getusersitepackages())') && mkdir -p /a/mm/ab/nn && cp -rn $site_pkg/ab/nn/* /a/mm/ab/nn/ || true && export RUN_TS=$(date +%Y-%m-%d_%H-%M-%S) && mkdir -p /a/mm/ab/gpt/brute/ga/meta_evolution/logs && python3 -u -m ab.gpt.brute.ga.meta_evolution.meta_evolver 2>&1 | tee /a/mm/ab/gpt/brute/ga/meta_evolution/logs/pod_${RUN_TS}.log"
```

### Why This Line Is Critical

This single line is the **entire container startup sequence**. It runs inside a Kubernetes pod and performs the following steps in strict order:

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `cd /a/mm` | Navigate to the bind-mounted codebase root |
| 2 | `pip uninstall -y flash-attn` | Remove flash-attn (causes PermissionError in container) |
| 3 | `pip install git+...nn-dataset --upgrade` | Install/upgrade the `nn-dataset` package (provides `ab.nn`) |
| 4 | `export PYTHONPATH=/a/mm` | Set Python import root to the mounted codebase |
| 5 | `site_pkg=$(python3 -c '...')` | Detect the user site-packages path |
| 6 | `mkdir -p /a/mm/ab/nn` | Ensure the `ab/nn` directory exists |
| 7 | `cp -rn $site_pkg/ab/nn/* /a/mm/ab/nn/ \|\| true` | Copy NEW pip files into local codebase (no-clobber) |
| 8 | `export RUN_TS=$(date +%Y-%m-%d_%H-%M-%S)` | Timestamp for log file naming |
| 9 | `mkdir -p .../logs` | Ensure logs directory exists |
| 10 | `python3 -u -m ab.gpt.brute.ga.meta_evolution.meta_evolver 2>&1 \| tee ...` | Run the meta-evolver with output logging |

### Consequences of Modifying This Line

- **Breaking the `&&` chain:** If any step fails, all subsequent steps are skipped. The pod will exit with an error and no training will occur.
- **Removing `pip uninstall flash-attn`:** Will cause `PermissionError` during model loading.
- **Changing `PYTHONPATH`:** All imports (`ab.gpt.*`, `ab.nn.*`) will fail with `ModuleNotFoundError`.
- **Altering `cp -rn`:** May overwrite local code with stale pip versions, or leave `ab.nn` incomplete.
- **Changing the module path:** The meta-evolver won't start.

### Rule

> **NEVER modify this line unless the user explicitly requests it and confirms the exact change.**  
> If a proposed change requires altering this line, STOP and inform the user before proceeding.

---

## 📋 Full File Analysis

### Section 1: Job Identity (Lines 1–6)

```json
"name": "nngpt-fractal-meta-evo-clone3"
```

- **Importance:** This is the Kubernetes job name used in all `kubectl` commands.
- **Must be unique** per clone to avoid conflicts with other running jobs.
- **Safe to change** when deploying from a different clone directory.

### Section 2: Container Image (Line 15)

```json
"image": "abrainone/ai-linux:cu12.6.3-latest"
```

- Pre-built image with Python 3.12, PyTorch, CUDA 12.6.3.
- `imagePullPolicy: Always` ensures the latest image tag is used.
- **Caution:** Changing the image may break compatibility with the startup command chain.

### Section 3: Environment Variables (Lines 24–41)

| Variable | Value | Purpose |
|----------|-------|---------|
| `POPULATION_SIZE` | `20` | Number of individuals per GA generation |
| `GENERATIONS` | `5` | Number of GA generations per benchmark |
| `MUTATION_RATE` | `0.6` | Probability of gene mutation |
| `META_ATTEMPTS` | `5` | Number of meta-evolution iterations |

- **Safe to modify** for tuning experiments.
- Changes take effect on the next pod deployment (`kubectl apply`).

### Section 4: Security Context (Lines 42–46)

```json
"runAsUser": 1017,
"runAsGroup": 1376
```

- Runs as user `b-a-singh` (UID 1017) inside the container.
- **Do not change** — must match the host filesystem permissions for `/shared/ssd/home/b-a-singh/`.

### Section 5: Resources (Lines 47–58)

| Resource | Request | Limit |
|----------|---------|-------|
| GPU | 1 | 1 |
| CPU | 10 | 16 |
| Memory | 32Gi | 64Gi |

- **Safe to adjust** within cluster quota limits.
- Reducing memory below 32Gi risks OOM kills during training.

### Section 6: Volume Mounts (Lines 59–72)

| Container Path | Host Path | Purpose |
|----------------|-----------|---------|
| `/a/mm` | `/shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt` | Main codebase |
| `/a/mm/data` | `/shared/local/data/a-group-automl/data` | Shared datasets (CIFAR-10, etc.) |
| `/dev/shm` | *emptyDir (Memory, 16Gi)* | Shared memory for DataLoader workers |

- **CRITICAL:** The `v0` hostPath (line 87) **must match the clone directory**. Using the wrong path causes the container to run stale/wrong code.
- The `v1` data path is shared across all users — **never modify or delete** files under this mount.

---

## 🚀 Deployment Quick Reference

### Deploy / Redeploy
```bash
kubectl delete job nngpt-fractal-meta-evo-clone3 --ignore-not-found \
  && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt.json
```

### Monitor Logs (live)
```bash
kubectl logs -f -l job-name=nngpt-fractal-meta-evo-clone3
```

### Check Job Status
```bash
kubectl get jobs nngpt-fractal-meta-evo-clone3
kubectl describe job nngpt-fractal-meta-evo-clone3
```

---

## 📝 Change Log

| Date | Change | By |
|------|--------|----|
| 2026-05-06 | Document created with critical rules and full analysis | Agent |
| 2026-04-26 | Job name changed from `nngpt-fractal-meta-evo-01` to `nngpt-fractal-meta-evo-clone3` | User |
| 2026-05-03 | Added `RUN_TS` timestamp and `tee` logging to args line | User |
