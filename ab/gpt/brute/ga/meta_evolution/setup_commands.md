# Meta-Evolution Job Commands

This file contains the common commands needed to run the Meta-Evolution and Meta-Baseline jobs, as well as how to view their live logs.

## 1. Meta-Evolution (LLM-Guided GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-fractal-meta-evo-clone3-cifar10
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt.json
```

**To delete existing .pkl files (force fresh restart of GA population):**
```bash
rm -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GenFractal_ckpt_cifar10.pkl
rm -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GenFractal_ckpt.pkl
```

**To delete existing LLM fine-tuned weights (force fresh restart of LLM's memory):**
```bash
rm -rf /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/fine_tuned_adapter_cifar10
rm -rf /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/fine_tuned_adapter
```

---

## 2. Meta-Baseline (Standard GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GA_run_baseline.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GA_run_baseline.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-baseline-ga-benchmark
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GA_run_baseline.json
```

---

## 3. Meta-Evolution CIFAR-100 (LLM-Guided GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-fractal-meta-evo-clone3-cifar100
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100.json
```

**To delete existing .pkl files:**
```bash
rm /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GenFractal_ckpt_cifar100.pkl
```

---

## 4. Meta-Baseline CIFAR-100 (Standard GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GA_run_baseline_cifar100.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GA_run_baseline_cifar100.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-baseline-ga-benchmark-cifar100
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/GA_run_baseline_cifar100.json
```

---

## 5. General Kubernetes Commands

**To see all currently running pods (to check their status):**
```bash
kubectl get pods
```

**To see all active jobs:**
```bash
kubectl get jobs
```

---

## 6. Resetting Checkpoints

If you need to start fresh, you can clear the existing `.pkl` checkpoint/state files from the workspace.

**To delete existing .pkl files:**
```bash
rm /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/fractal_baseline_save_point.pkl
rm /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/fractal_baseline_save_point_cifar100.pkl
```
