"""STEP 2: GPU-vs-CPU timing benchmark on a few conditions.

The nets are tiny (3->64->64->27) but the VDN runner batches ALL (rep x agent)
nets and the whole env loop into single tensor ops, so GPU may still win. We
measure, we don't assume. Times a handful of conditions at reduced episodes on:
GPU0 (7900 XTX), GPU1 (7800 XT), and CPU. Reuses the EXISTING b_vdn_runner
machinery, swapping only the preference sampler (monkeypatch).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import results.review_upgrade.b_vdn_runner as vr  # noqa: E402
from results.adversarial_rerun.dgp import sample_agent_params_equicorr  # noqa: E402
from friction_marl.experiments.factorial_design import build_action_map  # noqa: E402

# monkeypatch: rho passed in place of alpha
vr.sample_agent_params = lambda rng, A, M, rho, sigma: sample_agent_params_equicorr(rng, A, M, rho, sigma)

N_REPS = 30
N_EP_BENCH = 100          # 1/10 of full 1000-ep run; multiply by 10 to extrapolate
CONDS = [(-0.33, 1.0, "vdn"), (0.0, 1.0, "iql"), (0.8, 0.6, "vdn")]


def bench_device(devstr):
    if devstr == "cpu":
        device = torch.device("cpu")
        name = "CPU (9900X)"
    else:
        idx = int(devstr)
        device = torch.device(f"cuda:{idx}")
        name = f"GPU{idx} ({torch.cuda.get_device_name(idx)})"
    action_map = build_action_map(vr.N_RESOURCES)
    amap = torch.tensor(action_map, device=device)
    probe = torch.tensor(
        np.random.default_rng(0).uniform(0, vr.CAPACITY, size=(128, vr.N_RESOURCES)).astype(np.float32),
        device=device)
    # warmup (let allocator / kernels compile)
    vr.run_condition(0.0, 0.6, 0.0, "vdn", 4, 5, device, 1, amap, probe)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    per = []
    for (rho, sig, mode) in CONDS:
        t0 = time.time()
        vr.run_condition(rho, sig, 0.0, mode, N_REPS, N_EP_BENCH, device, 123, amap, probe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per.append(time.time() - t0)
    return name, per


def main():
    print(f"=== GPU vs CPU benchmark: {N_REPS} reps x {N_EP_BENCH} ep, conds={CONDS} ===")
    results = {}
    for dev in ["0", "1", "cpu"]:
        try:
            name, per = bench_device(dev)
            mean_s = float(np.mean(per))
            full_est = mean_s * (1000 / N_EP_BENCH)
            results[dev] = (name, mean_s, full_est)
            print(f"  {name:<28} per-cond {N_EP_BENCH}ep: {mean_s:6.1f}s  "
                  f"=> full(1000ep) ~{full_est:6.1f}s   raw={[f'{x:.1f}' for x in per]}")
        except Exception as e:
            print(f"  device {dev}: FAILED ({type(e).__name__}: {e})")
    if "0" in results and "cpu" in results:
        g = results["0"][2]; c = results["cpu"][2]
        print(f"\n  GPU0 vs CPU speedup (full-run estimate): {c/g:.2f}x")
    # pick fastest
    best = min(results, key=lambda k: results[k][1]) if results else None
    if best:
        print(f"  FASTEST: {results[best][0]}  (full-run ~{results[best][2]:.0f}s/cond)")
    return results


if __name__ == "__main__":
    main()
