# 1. Parallel Word Counting in a Text Corpus with MPI

**Lab 2 – Estructura del Computador II**
**Student:** Santy Baza Perez

## 2. Problem Description

A synthetic corpus of 3,000 text files (`dataset/file_XXXX.txt`, 44,951,458 tokens, ≈275 MB) and a query file (`dataset/consulta.txt`, 200 distinct words) are generated with the provided `generator.py` (seed 42, Zipf-like weighted vocabulary). The task is to count how many times each query word appears in the whole corpus and print the 10 most frequent ones.

File sizes are deliberately heterogeneous (60% small, 30% medium, 10% large, ranging from 2,000 to 100,000 words), so assigning the same *number* of files to every process does not assign the same *amount of work*. The lab compares:

- the provided sequential baseline,
- MPI version 1: query broadcast + static block distribution of files,
- MPI version 2: a distribution that reduces the resulting load imbalance.

### Repository contents

| File | Description |
|---|---|
| `generator.py` | Dataset generator (as provided, unchanged) |
| `baseline_secuencial.py` | Sequential baseline (as provided, unchanged) |
| `run_all.sh` | Provided batch script (as provided, unchanged) |
| `detectar_procesos_mpi.py` | Provided process-count detector (as provided, unchanged) |
| `mpi1.py` | MPI version 1 (static block distribution) |
| `mpi2.py` | MPI version 2 (size-aware LPT distribution) |
| `experiments/run_experiments.sh` | Runs the full experimental plan inside the container |
| `experiments/summarize.py` | Parses logs and computes T_p, S_p, E_p and LI |
| `experiments/results/` | Main experiment (p = 14): logs, `runs.csv`, `per_rank.csv`, `summary.md` |
| `experiments/results_p8/` | Supplementary experiment (p = 8), same format |

## 3. Environment and Execution Instructions

**Hardware / software used for all measurements**

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5700X, 8 physical cores / 16 hardware threads (SMT) |
| Memory visible to the container | 16 GB |
| Host | Windows 11 Pro + Docker Desktop (WSL2 backend) |
| Image | `augustosalazar/slim-mpi:2` (Python 3.8.20, mpi4py 4.1.1, Open MPI 4.1.4) |
| `nproc` inside the container | 16 |

**Commands** (run from the repository root; on Windows `cmd` replace `"$(pwd)"` with `"%cd%"`):

```bash
# Generate the dataset
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py

# Sequential baseline
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# Process-count detection
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py

# MPI versions (p = 14 on this machine)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 14 python /app/mpi1.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 14 python /app/mpi2.py

# Full experimental plan used for this report (generation, detection, 5 runs of each program, summary)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/experiments/run_experiments.sh
# Optional variables: RUNS (default 5), NP (default: detected), OUT (output folder)
docker run --rm -e NP=8 -e OUT=/app/experiments/results_p8 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/experiments/run_experiments.sh

# Provided batch script (also works with mpi1.py / mpi2.py, which print EXECUTION_TIME=...)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

### 3.1 Experimental Plan

1. Generate the dataset once with `generator.py` (deterministic, seed 42).
2. Determine p with `detectar_procesos_mpi.py`: it times a CPU-bound MPI task for N = 1…16 and recommends the fastest N.
3. Run the baseline, MPI v1 and MPI v2 **5 times each** with that p. The three programs are interleaved inside each run (baseline → v1 → v2) so that slow drifts of the machine affect all of them equally.
4. For every MPI execution record the total time (`EXECUTION_TIME`, measured by rank 0 from a barrier after `MPI_Init` until the global top 10 is built), the wall time of the whole `mpirun` command, and the local processing time t_i of every rank.
5. Compute T_s and T_p as the mean over runs, S_p = T_s / T_p, E_p = S_p / p, LI = max(t_i) / mean(t_i) (averaged over runs) and the relative spread (max − min) / mean.
6. Verify correctness: every MPI run compares its full per-word table against `baseline_results.csv` and its top 10 against the baseline.
7. Supplementary run with p = 8 (one process per physical core) to check whether the conclusions depend on SMT oversubscription.

`run_experiments.sh` copies the code into the container file system (`/work`) and generates the dataset there, so that reading 275 MB through the Windows bind mount does not dominate the measurements. Only logs are written back to the repository.

### 3.2 Sequential baseline (what it does and how)

`baseline_secuencial.py` loads `consulta.txt` into a set (lowercase), iterates over every `file_*.txt` in the dataset, splits each line on whitespace, lowercases every token, and increments a `Counter` when the token is in the query set. It reports the elapsed time of the counting phase (`time.perf_counter`), the number of files and tokens, saves the full table to `dataset/baseline_results.csv` and prints the top 10.

### 3.3 MPI version 1 (what it does and how)

`mpi1.py`:

1. Rank 0 reads `consulta.txt`.
2. The query words are broadcast to all ranks with `comm.bcast`.
3. Rank 0 lists the `file_*.txt` files (sorted by name).
4. The list is split into p contiguous blocks of ⌈3000/p⌉ or ⌊3000/p⌋ files and sent with `comm.scatter` (static distribution by file count).
5. Each rank counts the query words in its files using exactly the same tokenization rule as the baseline and measures its local processing time t_i with `MPI.Wtime`.
6. Partial `Counter`s, file counts, bytes, tokens and t_i are collected in rank 0 with `comm.gather`.
7. Rank 0 merges the counters, prints a per-process table (files, MB, tokens, local time), LI, spread, whether the result matches `baseline_results.csv`, the top 10, and `EXECUTION_TIME=<seconds>`.

### 3.4 Test procedure

- **Correctness:** each MPI execution compares its complete word → count table with `baseline_results.csv` (written by the baseline in the same run) and prints `Matches sequential baseline: yes/NO`. `summarize.py` also checks that the top 10 is identical across all runs and programs.
- **Timing:** 5 repetitions per program with the same p; mean, standard deviation and per-run values are reported. All executions run on a warm page cache (the dataset was just written), equally for all programs.
- **Load balance:** the per-rank local times of every run are stored in `per_rank.csv`; LI and spread are computed per run and then averaged.

## 4. Experimental Plan Execution

**Process count.** `detectar_procesos_mpi.py` recommended **p = 14** (median times, s):

| N | 1 | 2 | 4 | 8 | 11 | 12 | 13 | **14** | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| median (s) | 4.005 | 2.291 | 1.451 | 1.170 | 1.118 | 1.252 | 1.055 | **1.039** | 1.452 | 1.152 |

**Correctness.** All 10 MPI executions at p = 14 (and all 10 at p = 8) produced exactly the same full table as the baseline. Top 10: a 785,774 · para 392,156 · sus 228,913 · otros 105,530 · ante 99,832 · unos 88,794 · otra 83,901 · vosotros 61,617 · mios 58,420 · tuya 56,635 (total 3,631,778 occurrences).

### 4.1 Sequential baseline timing

| Run | 1 | 2 | 3 | 4 | 5 | **Mean ± sd** |
|---|---|---|---|---|---|---|
| T (s) | 7.083 | 7.239 | 7.124 | 7.436 | 7.439 | **T_s = 7.264 ± 0.168** |

Throughput: 44.95 M tokens / 7.264 s ≈ 6.2 M tokens/s on one core.

### 4.2 MPI version 1 timing results (p = 14)

| Run | T_p (s) | Wall incl. `mpirun` (s) | max t_i | mean t_i | min t_i | LI | Spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.404 | 1.860 | 1.381 | 1.229 | 1.015 | 1.124 | 0.298 |
| 2 | 1.403 | 1.870 | 1.380 | 1.183 | 0.980 | 1.166 | 0.338 |
| 3 | 1.440 | 1.908 | 1.414 | 1.266 | 1.116 | 1.116 | 0.235 |
| 4 | 1.555 | 2.073 | 1.530 | 1.352 | 1.120 | 1.132 | 0.303 |
| 5 | 1.566 | 2.038 | 1.540 | 1.377 | 1.178 | 1.119 | 0.263 |
| **Mean** | **1.474 ± 0.081** | 1.950 | 1.449 | 1.282 | 1.082 | **1.131** | 0.287 |

**S_14 = 7.264 / 1.474 = 4.93**, **E_14 = 4.93 / 14 = 0.35**. Including process launch (wall time), the speedup is 7.292 / 1.950 = 3.74.

### 4.3 Load imbalance evidence (per-process local times and LI)

Version 1, mean local time over the 5 runs:

| Rank | Files | MB | Tokens | Mean t_i (s) |
|---:|---:|---:|---:|---:|
| 0 | 215 | 21.41 | 3,499,659 | 1.368 |
| 1 | 215 | 20.60 | 3,365,828 | 1.340 |
| 2 | 215 | 16.66 | 2,723,658 | 1.087 |
| 3 | 215 | 19.16 | 3,129,794 | 1.262 |
| 4 | 214 | 21.59 | 3,527,804 | 1.384 |
| 5 | 214 | 21.17 | 3,459,640 | 1.378 |
| 6 | 214 | 20.02 | 3,273,088 | 1.366 |
| 7 | 214 | 17.69 | 2,890,577 | 1.155 |
| 8 | 214 | 20.60 | 3,365,929 | 1.301 |
| 9 | 214 | 17.63 | 2,881,056 | 1.149 |
| 10 | 214 | 21.13 | 3,454,443 | 1.355 |
| 11 | 214 | 18.33 | 2,995,511 | 1.193 |
| 12 | 214 | 16.96 | 2,770,904 | 1.162 |
| 13 | 214 | 22.11 | 3,613,567 | 1.442 |

Every rank gets 214–215 files, but the data volume ranges from 16.66 MB (rank 2) to 22.11 MB (rank 13), a 1.33× difference, and local time follows it (1.087 s vs 1.442 s). The imbalance is systematic: rank 13 (most data) was the slowest in 4 of 5 runs and rank 2 (least data) was among the three fastest in all 5. Mean **LI = 1.131** (1.116–1.166) and mean spread = 0.287, so on average the slowest rank works 13% longer than the mean while the fastest ones idle for ≈0.37 s.

### 4.4 MPI version 2: how it corrects the imbalance, and its results

**Strategy.** The local time is proportional to the number of tokens, which is proportional to the file size. In `mpi2.py`, rank 0 reads the size of every file (`os.path.getsize`), sorts the files by size in descending order and assigns each one to the rank with the smallest accumulated number of bytes, using a min-heap (*Longest Processing Time first* greedy). The assignment is sent with `comm.scatter`; everything else (bcast of the query, counting, gather, verification) is identical to version 1.

Result: every rank receives 19.64–19.65 MB (≈3.21 M tokens) instead of 16.66–22.11 MB.

| Run | T_p (s) | Wall incl. `mpirun` (s) | max t_i | mean t_i | min t_i | LI | Spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.433 | 1.896 | 1.387 | 1.332 | 1.284 | 1.042 | 0.078 |
| 2 | 1.486 | 1.946 | 1.428 | 1.323 | 1.267 | 1.079 | 0.122 |
| 3 | 1.742 | 2.245 | 1.680 | 1.443 | 1.328 | 1.164 | 0.244 |
| 4 | 1.530 | 1.995 | 1.480 | 1.401 | 1.319 | 1.056 | 0.115 |
| 5 | 1.482 | 1.970 | 1.432 | 1.337 | 1.275 | 1.071 | 0.118 |
| **Mean** | **1.535 ± 0.121** | 2.010 | 1.481 | 1.367 | 1.295 | **1.082** | 0.135 |

Per-rank mean local times range from 1.321 s to 1.460 s (full table in `experiments/results/summary.md`); the residual imbalance is no longer tied to data size but to runtime noise (e.g. run 3, where ranks 0 and 8 were slowed down).

**Comparison (p = 14)**

| Program | Mean T (s) | Speedup S_p | Efficiency E_p | Mean LI | Spread | Mean wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.264 | 1.00 | 1.00 | – | – | 7.292 |
| MPI v1 (block) | 1.474 | 4.93 | 0.35 | 1.131 | 0.287 | 1.950 |
| MPI v2 (LPT) | 1.535 | 4.73 | 0.34 | 1.082 | 0.135 | 2.010 |

**Supplementary experiment (p = 8, one process per physical core, 5 runs each)** – `experiments/results_p8/`

| Program | Mean T (s) | Speedup S_p | Efficiency E_p | Mean LI | Spread |
|---|---:|---:|---:|---:|---:|
| Baseline | 7.462 ± 0.504 | 1.00 | 1.00 | – | – |
| MPI v1 (block) | 1.824 ± 0.058 | 4.09 | 0.51 | 1.122 | 0.289 |
| MPI v2 (LPT) | 1.659 ± 0.161 | 4.50 | 0.56 | 1.053 | 0.091 |

## 5. Analysis

### 5.1 Parallel performance

Yes. MPI v1 reduced the time from 7.264 s to 1.474 s (S_14 = 4.93), but it is far from the ideal S_p = 14 (E_14 = 0.35). The gap can be decomposed with the measured times:

- **Communication and serial parts are small:** T_p − max(t_i) = 1.474 − 1.449 ≈ 0.025 s (≈2% of T_p). Broadcasting 200 words, scattering file names and gathering 14 counters of ≤200 entries is negligible.
- **Load imbalance costs ≈11%:** with perfect balance the time would be close to mean(t_i) = 1.282 s instead of max(t_i) = 1.449 s (S ≈ 5.7 instead of 4.9).
- **The dominant factor is the hardware:** each rank processes ≈3.2 M tokens in ≈1.28 s (≈2.5 M tokens/s), 2.5× slower than the sequential program (6.2 M tokens/s). Fourteen processes run on only 8 physical cores, so pairs share a core through SMT, the CPU runs at a lower all-core clock than in single-thread boost, and L3 cache and memory bandwidth are shared. The aggregate throughput (≈30 M tokens/s) is therefore bounded by about what 8 cores can deliver, not 14. The detector shows the same effect for a pure CPU task: N = 8 gives only 3.4× over N = 1, and N = 14 only 3.9×.
- **Launch overhead outside the measured region:** starting `mpirun`, 14 Python interpreters and mpi4py adds ≈0.48 s (wall 1.950 s vs 1.474 s), lowering the end-to-end speedup to 3.74.

### 5.2 Load balance

Version 1 showed clear, reproducible imbalance: identical file counts (214–215) but 16.66–22.11 MB per rank, local times from 1.087 s to 1.442 s, mean LI = 1.131 and spread = 0.287. The slowest ranks on average were those with the most data (ranks 13, 4, 5, 0).

Version 2 equalized the data volume (19.64–19.65 MB per rank) and halved the spread (0.287 → 0.135), lowering LI from 1.131 to 1.082. **At p = 14 this did not produce a real performance improvement:** T_p went from 1.474 s to 1.535 s, a difference smaller than the run-to-run variation (sd 0.081 and 0.121 s; v2's median, 1.486 s, is close to v1's, 1.440 s). The reason is visible in the mean local time, which increased from 1.282 s to 1.367 s. In v1, the light ranks finish early and free hardware threads, so the heavy ranks speed up during the last part of the execution. In v2, all 14 ranks compete for the 8 cores until the end. When the machine is saturated, the critical path is set by total work divided by aggregate throughput, and redistributing the work does not change that.

The supplementary run with p = 8 supports this explanation. Without SMT sharing, the same LPT strategy lowered LI from 1.122 to 1.053 and T_p from 1.824 s to 1.659 s (−9%, S: 4.09 → 4.50, E: 0.51 → 0.56). With only 5 runs and one slow v2 run (1.887 s), this improvement is moderate evidence, not conclusive.

### 5.3 Limitations

- **Few repetitions and noisy timings:** 5 runs per program, on a desktop Windows host where background activity is not controlled (baseline sd 0.17 s at p = 14 and 0.50 s at p = 8, with one 8.36 s outlier). Differences under ≈0.15 s between v1 and v2 are not significant, which weakens the comparison between versions more than the comparison with the baseline.
- **Virtualized environment:** Docker Desktop runs inside a WSL2 VM, so the scheduler, CPU frequency and SMT placement are not controlled (no process pinning). Results are specific to this machine.
- **p chosen by a noisy CPU-bound test:** the detector curve is not monotonic (e.g. N = 13: 1.055 s, N = 14: 1.039 s, N = 15: 1.452 s), so p = 14 is only marginally better than its neighbours, and it oversubscribes the physical cores. This is why a p = 8 run was added.
- **Warm cache and in-container file system:** the dataset is in the page cache and on the container's file system. With the dataset on the Windows bind mount, `run_all.sh` at p = 2 took 13.7–15.8 s (mostly file-system latency), so I/O-bound scenarios would behave differently.
- **Single dataset and single size model:** imbalance comes only from file size (deterministic seed 42). LPT uses bytes as the cost model, which holds here but would not cover runtime variations such as a slower node; a dynamic (work-queue) distribution would handle those.

Overall, the conclusion that MPI improves the baseline by ≈4.9× is robust (all 10 MPI runs at p = 14 were more than 4× faster than the baseline mean). The conclusion about the benefit of v2 depends on p and is only moderately supported.

## 6. Conclusions

Both MPI versions produce exactly the same counts as the sequential baseline and are about 5× faster (S_14 = 4.93 for v1 and 4.73 for v2, E ≈ 0.35). Efficiency is limited mainly by running 14 processes on 8 physical cores, not by communication (≈2% of T_p) or imbalance (≈11%). I would recommend the size-aware LPT distribution (v2): it costs only a stat call per file, removes the systematic imbalance (LI 1.131 → 1.082 at p = 14, 1.122 → 1.053 at p = 8) and, when each process has its own physical core (p = 8), reduced the execution time by 9% (S 4.09 → 4.50, E 0.51 → 0.56). At the oversubscribed p = 14 its time was statistically indistinguishable from the static block distribution, so its benefit on this machine shows up only when p does not exceed the number of physical cores.
