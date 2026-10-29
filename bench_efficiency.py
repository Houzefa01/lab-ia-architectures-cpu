"""
bench_efficiency.py — reusable CPU inference efficiency benchmark.

For a fixed batch size (default 32) and a list of sequence lengths, measures
per length:
  - latency_ms          : median of 5 timed forward passes (torch.no_grad,
                           model.eval()), after 5 untimed warmup passes.
                           Timed with time.perf_counter.
  - peak_ram_mb          : peak *process* RSS delta around the forward-pass
                           loop, via resource.getrusage(RUSAGE_SELF).ru_maxrss
                           (the rigorous number — captures the real C++/ATen
                           tensor allocator memory that Python-level tools miss).
  - tracemalloc_peak_mb  : peak memory tracked by tracemalloc over the same
                           window. NOTE: torch tensors are allocated by the
                           C++ allocator, not CPython's, so tracemalloc only
                           sees Python-object overhead here and will typically
                           read near-zero / uninformative for large tensors.
                           Reported anyway because the spec asked for it —
                           treat peak_ram_mb (RSS-based) as the trustworthy
                           number, not this one.
  - attn_matrix_theoretical_mb : theoretical size in MB of a single
                           (batch, heads, T, T) float32 attention matrix —
                           i.e. what a full softmax-attention layer must
                           materialize at this (batch, T). Reference only:
                           it does NOT assert the model under test actually
                           allocates a tensor of this shape.

Each length is measured in its own forked subprocess so that:
  - a single run exceeding `timeout_s` (default 180s / 3min) can be detected
    and the whole length marked "timeout" — remaining (larger) lengths are
    then skipped by the caller, without ever corrupting a later length's
    memory baseline with the previous length's still-resident allocations;
  - an OOM — either a caught RuntimeError from the CPU allocator, or the OS
    OOM-killer SIGKILL-ing the subprocess outright — is caught/detected and
    recorded as "oom", WITHOUT crashing the parent process or script.

Usage:
    import functools
    from bench_efficiency import bench
    from baseline.transformer import BaselineTransformer

    factory = functools.partial(BaselineTransformer, vocab_size=65, dim=64,
                                 heads=2, layers=3, ffn_dim=256,
                                 max_seq_len=4096)
    result = bench(factory, "BaselineTransformer")

`model_factory` must be a zero-arg callable returning a fresh nn.Module with
a `.vocab_size` attribute and the project's standard
`(B, T) -> (B, T, vocab_size)` forward interface (functools.partial around a
model class works well). Subprocess isolation uses the `fork` start method
(Linux/macOS only — fine for this project's CPU-only environment), so plain
lambdas/closures are also fine as `model_factory`.
"""

import functools
import json
import multiprocessing as mp
import os
import resource
import time
import tracemalloc

import torch

DEFAULT_LENGTHS = [48, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_BATCH = 32
DEFAULT_TIMEOUT_S = 180.0  # single-run abort threshold (spec: "3 min CPU")
N_WARMUP = 5
N_TIMED = 5


def _theoretical_attn_mb(batch, heads, T, dtype_bytes=4):
    """Size in MB of one (batch, heads, T, T) dtype_bytes-wide attention matrix."""
    return batch * heads * T * T * dtype_bytes / 1e6


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _worker(model_factory, T, batch_size, heads, timeout_s, q):
    """Runs in a forked subprocess: build model, warm up, time, measure memory."""
    try:
        torch.manual_seed(0)
        model = model_factory()
        model.eval()
        vocab_size = getattr(model, "vocab_size", None)
        if vocab_size is None:
            raise AttributeError(
                "model has no .vocab_size attribute — required by bench_efficiency"
            )
        x = torch.randint(0, vocab_size, (batch_size, T), dtype=torch.long)

        theoretical_mb = _theoretical_attn_mb(batch_size, heads, T)

        tracemalloc.start()
        ru_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB (Linux)

        latencies = []
        timed_out = False
        last_dt = None
        with torch.no_grad():
            for _ in range(N_WARMUP):
                t0 = time.perf_counter()
                _ = model(x)
                last_dt = time.perf_counter() - t0
                if last_dt > timeout_s:
                    timed_out = True
                    break
            if not timed_out:
                for _ in range(N_TIMED):
                    t0 = time.perf_counter()
                    _ = model(x)
                    last_dt = time.perf_counter() - t0
                    latencies.append(last_dt)
                    if last_dt > timeout_s:
                        timed_out = True
                        break

        _, trace_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ru_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if timed_out:
            q.put({
                "status": "timeout",
                "attn_matrix_theoretical_mb": theoretical_mb,
                "last_run_s": last_dt,
            })
            return

        latencies_ms = [l * 1000.0 for l in latencies]
        q.put({
            "status": "ok",
            "latency_ms": _median(latencies_ms),
            "latency_ms_all": latencies_ms,
            "peak_ram_mb": max(0.0, (ru_after - ru_before) / 1024.0),  # KB -> MB
            "tracemalloc_peak_mb": trace_peak / 1e6,
            "attn_matrix_theoretical_mb": theoretical_mb,
        })
    except (RuntimeError, MemoryError) as e:
        q.put({"status": "oom", "error": str(e)[:300]})
    except Exception as e:  # noqa: BLE001 - never let a per-length failure escape
        q.put({"status": "error", "error": f"{type(e).__name__}: {e}"[:300]})


def _run_one_length(model_factory, T, batch_size, heads, timeout_s):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(model_factory, T, batch_size, heads, timeout_s, q))
    p.start()
    # Safety-net wall clock cap: the worker self-aborts as soon as ONE run
    # exceeds timeout_s, so in practice it returns almost immediately after
    # that. This cap only protects against a genuinely hung/OOM-thrashing
    # subprocess that never gets back to Python to report anything.
    hard_cap = timeout_s + 90.0
    p.join(hard_cap)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "timeout", "reason": "hard wall-clock cap exceeded (subprocess hung)"}
    try:
        result = q.get_nowait()
    except Exception:
        result = None
    if result is None:
        # Subprocess exited without ever putting a result -> most likely
        # SIGKILL'd by the OS OOM-killer.
        return {"status": "oom", "reason": f"subprocess exitcode={p.exitcode} (likely OOM-killed)"}
    return result


def bench(model_factory, name, lengths=None, batch_size=DEFAULT_BATCH, heads=2,
          timeout_s=DEFAULT_TIMEOUT_S, verbose=True):
    """Benchmark inference latency + peak memory of `model_factory()` across `lengths`.

    Returns: {"model": name, "device": "cpu", "threads": <n>, "batch_size": <n>,
              "lengths": {"<T>": {...}, ...}}

    Never raises on a per-length timeout/OOM: that length (and all longer
    ones) are recorded with status "timeout"/"oom"/"skipped" instead.
    """
    lengths = list(lengths) if lengths is not None else DEFAULT_LENGTHS
    out = {
        "model": name,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "batch_size": batch_size,
        "lengths": {},
    }
    aborted = False
    for T in lengths:
        if aborted:
            out["lengths"][str(T)] = {"status": "skipped", "reason": "earlier length timed out / OOMed"}
            continue
        if verbose:
            print(f"[bench_efficiency] {name}: T={T} ...", flush=True)
        try:
            res = _run_one_length(model_factory, T, batch_size, heads, timeout_s)
        except Exception as e:  # noqa: BLE001 - the whole script must never crash
            res = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        out["lengths"][str(T)] = res
        if verbose:
            if res["status"] == "ok":
                print(
                    f"  -> latency={res['latency_ms']:.2f}ms  peak_ram={res['peak_ram_mb']:.1f}MB "
                    f"(theoretical attn matrix={res['attn_matrix_theoretical_mb']:.1f}MB)",
                    flush=True,
                )
            else:
                print(f"  -> {res['status']}: {res.get('reason', res.get('error', ''))}", flush=True)
        if res["status"] in ("timeout", "oom", "error"):
            aborted = True
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run bench_efficiency on BaselineTransformer")
    parser.add_argument("--out", default="resultats/baseline_ref.json")
    parser.add_argument("--vocab-size", type=int, default=65, help="65 = TinyShakespeare char vocab")
    args = parser.parse_args()

    from baseline.transformer import BaselineTransformer

    factory = functools.partial(
        BaselineTransformer,
        vocab_size=args.vocab_size, dim=64, heads=2, layers=3, ffn_dim=256,
        max_seq_len=4096,
    )
    result = bench(factory, "BaselineTransformer")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== Saved to", args.out, "===")
    print(json.dumps(result, indent=2))
