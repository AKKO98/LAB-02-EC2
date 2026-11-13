"""Parses the experiment logs and writes CSV tables plus summary.md."""

import csv
import glob
import os
import re
import statistics
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
LOGS = os.path.join(OUT, "logs")
NP = int(open(os.path.join(OUT, "np.txt")).read().strip())
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$")


def run_id(path):
    return int(re.search(r"run(\d+)\.log$", path).group(1))


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def top10(text):
    block = text.split("Top 10", 1)[1].splitlines()[1:]
    return [line.strip() for line in block if ":" in line][:10]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


wall = {}
with open(os.path.join(OUT, "wall_times.csv")) as f:
    for r in csv.DictReader(f):
        wall[(r["program"], int(r["run"]))] = float(r["wall_seconds"])

baseline = []
for path in sorted(glob.glob(os.path.join(LOGS, "baseline_run*.log")), key=run_id):
    text = read(path)
    t = float(re.search(r"Tiempo de ejecuci.n: ([\d.]+)", text).group(1))
    baseline.append({"run": run_id(path), "time": t, "wall": wall[("baseline", run_id(path))],
                     "top": top10(text)})

mpi = {}
per_rank = []
for prog in ("mpi1", "mpi2"):
    mpi[prog] = []
    for path in sorted(glob.glob(os.path.join(LOGS, f"{prog}_np{NP}_run*.log")), key=run_id):
        text = read(path)
        rid = run_id(path)
        ranks = []
        for line in text.splitlines():
            m = ROW.match(line)
            if m:
                ranks.append({"rank": int(m.group(1)), "files": int(m.group(2)),
                              "mb": float(m.group(3)), "tokens": int(m.group(4)),
                              "time": float(m.group(5))})
        times = [r["time"] for r in ranks]
        mt = mean(times)
        mpi[prog].append({
            "run": rid,
            "time": float(re.search(r"EXECUTION_TIME=([\d.]+)", text).group(1)),
            "wall": wall[(prog, rid)],
            "li": max(times) / mt,
            "spread": (max(times) - min(times)) / mt,
            "max": max(times), "mean": mt, "min": min(times),
            "match": re.search(r"Matches sequential baseline: (.+)", text).group(1).strip(),
            "top": top10(text),
        })
        for r in ranks:
            per_rank.append(dict(program=prog, run=rid, **r))

with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["program", "np", "run", "exec_time_s", "wall_time_s", "LI", "spread",
                "max_local_s", "mean_local_s", "min_local_s", "matches_baseline"])
    for b in baseline:
        w.writerow(["baseline", 1, b["run"], f"{b['time']:.6f}", f"{b['wall']:.6f}",
                    "", "", "", "", "", "reference"])
    for prog, runs in mpi.items():
        for r in runs:
            w.writerow([prog, NP, r["run"], f"{r['time']:.6f}", f"{r['wall']:.6f}",
                        f"{r['li']:.4f}", f"{r['spread']:.4f}", f"{r['max']:.6f}",
                        f"{r['mean']:.6f}", f"{r['min']:.6f}", r["match"]])

with open(os.path.join(OUT, "per_rank.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["program", "run", "rank", "files", "MB", "tokens", "local_time_s"])
    for r in per_rank:
        w.writerow([r["program"], r["run"], r["rank"], r["files"], f"{r['mb']:.2f}",
                    r["tokens"], f"{r['time']:.6f}"])

ts = mean([b["time"] for b in baseline])
ts_wall = mean([b["wall"] for b in baseline])
md = []
md.append(f"# Experiment summary (p = {NP}, {len(baseline)} runs per program)\n")

md.append("## Sequential baseline\n")
md.append("| Run | Time (s) | Wall incl. Python start (s) |")
md.append("|---:|---:|---:|")
for b in baseline:
    md.append(f"| {b['run']} | {b['time']:.3f} | {b['wall']:.3f} |")
md.append(f"| **Mean ± sd** | **{ts:.3f} ± {sd([b['time'] for b in baseline]):.3f}** | "
          f"{ts_wall:.3f} |\n")

for prog, title in (("mpi1", "MPI v1 (static block)"), ("mpi2", "MPI v2 (LPT by size)")):
    runs = mpi[prog]
    md.append(f"## {title}: per run\n")
    md.append("| Run | T_p (s) | Wall incl. mpirun (s) | max t_i | mean t_i | min t_i | LI | Spread | Matches baseline |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for r in runs:
        md.append(f"| {r['run']} | {r['time']:.3f} | {r['wall']:.3f} | {r['max']:.3f} | "
                  f"{r['mean']:.3f} | {r['min']:.3f} | {r['li']:.3f} | {r['spread']:.3f} | {r['match']} |")
    md.append(f"| **Mean** | **{mean([r['time'] for r in runs]):.3f}** | "
              f"{mean([r['wall'] for r in runs]):.3f} | {mean([r['max'] for r in runs]):.3f} | "
              f"{mean([r['mean'] for r in runs]):.3f} | {mean([r['min'] for r in runs]):.3f} | "
              f"**{mean([r['li'] for r in runs]):.3f}** | {mean([r['spread'] for r in runs]):.3f} | |\n")

    md.append(f"## {title}: per process (local time averaged over runs)\n")
    md.append("| Rank | Files | MB | Tokens | Mean t_i (s) | Min t_i | Max t_i |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for rank in range(NP):
        rows = [r for r in per_rank if r["program"] == prog and r["rank"] == rank]
        t = [r["time"] for r in rows]
        md.append(f"| {rank} | {rows[0]['files']} | {rows[0]['mb']:.2f} | {rows[0]['tokens']} | "
                  f"{mean(t):.3f} | {min(t):.3f} | {max(t):.3f} |")
    md.append("")

md.append("## Comparison\n")
md.append("| Program | Mean T (s) | sd (s) | Speedup S_p = T_s/T_p | Efficiency E_p = S_p/p | Mean LI | Mean wall (s) | Wall speedup |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
md.append(f"| Baseline | {ts:.3f} | {sd([b['time'] for b in baseline]):.3f} | 1.000 | 1.000 | – | {ts_wall:.3f} | 1.000 |")
for prog, name in (("mpi1", "MPI v1"), ("mpi2", "MPI v2")):
    runs = mpi[prog]
    tp = mean([r["time"] for r in runs])
    tw = mean([r["wall"] for r in runs])
    s = ts / tp
    md.append(f"| {name} (p={NP}) | {tp:.3f} | {sd([r['time'] for r in runs]):.3f} | {s:.3f} | "
              f"{s / NP:.3f} | {mean([r['li'] for r in runs]):.3f} | {tw:.3f} | {ts_wall / tw:.3f} |")
md.append("")

ref = baseline[0]["top"]
same = all(b["top"] == ref for b in baseline) and all(r["top"] == ref for v in mpi.values() for r in v)
md.append("## Correctness\n")
md.append(f"- Full per-word counts identical to `baseline_results.csv` in every MPI run: "
          f"{'yes' if all(r['match'] == 'yes' for v in mpi.values() for r in v) else 'NO'}")
md.append(f"- Top 10 identical across all runs and programs: {'yes' if same else 'NO'}\n")
md.append("| # | Word: count |")
md.append("|---:|---|")
for i, item in enumerate(ref, 1):
    md.append(f"| {i} | {item} |")

with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
