"""MPI version 2: query broadcast + size-aware (LPT) distribution of files."""

import heapq
import os
from collections import Counter

from mpi4py import MPI

TOP_N = 10
QUERY_NAME = "consulta.txt"
BASELINE_CSV = "baseline_results.csv"
OUTPUT_CSV = "mpi2_results.csv"


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        return sorted({line.strip().lower() for line in f if line.strip()})


def list_corpus(dataset_dir):
    return sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith("file_") and name.endswith(".txt")
        and os.path.isfile(os.path.join(dataset_dir, name))
    )


def lpt_partition(dataset_dir, files, parts):
    """Longest Processing Time first: largest file goes to the least loaded rank (load = bytes)."""
    sized = sorted(((os.path.getsize(os.path.join(dataset_dir, n)), n) for n in files),
                   key=lambda x: (-x[0], x[1]))
    heap = [(0, r) for r in range(parts)]
    chunks = [[] for _ in range(parts)]
    for nbytes, name in sized:
        load, r = heapq.heappop(heap)
        chunks[r].append(name)
        heapq.heappush(heap, (load + nbytes, r))
    return chunks


def count_files(dataset_dir, files, targets):
    """Same counting rule as baseline_secuencial.py (split on whitespace, lowercase)."""
    freq = Counter()
    tokens = 0
    for name in files:
        with open(os.path.join(dataset_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                words = [w.lower() for w in line.split()]
                tokens += len(words)
                for w in words:
                    if w in targets:
                        freq[w] += 1
    return freq, tokens


def save_csv(path, freq):
    with open(path, "w", encoding="utf-8") as f:
        f.write("palabra,conteo\n")
        for word in sorted(freq):
            f.write(f"{word},{freq[word]}\n")


def matches_baseline(path, freq):
    if not os.path.isfile(path):
        return "baseline CSV not found"
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        ref = {w: int(c) for w, c in (line.strip().split(",") for line in f if line.strip())}
    return "yes" if ref == dict(freq) else "NO"


def report(partials, total, elapsed, n_files, size, dataset_dir, label):
    times = [p["time"] for p in partials]
    mean_t = sum(times) / len(times)
    li = max(times) / mean_t if mean_t > 0 else float("nan")
    spread = (max(times) - min(times)) / mean_t if mean_t > 0 else float("nan")

    print(f"{label} | processes: {size} | files: {n_files}")
    print(f"{'rank':>4} {'files':>6} {'MB':>8} {'tokens':>10} {'local_time_s':>13}")
    for p in partials:
        print(f"{p['rank']:>4} {p['files']:>6} {p['bytes'] / 1e6:>8.2f} "
              f"{p['tokens']:>10} {p['time']:>13.6f}")
    print(f"LI (max/mean local time): {li:.4f}")
    print(f"Relative spread ((max-min)/mean): {spread:.4f}")
    print(f"Total tokens: {sum(p['tokens'] for p in partials)}")
    print(f"Total occurrences: {sum(total.values())}")
    print(f"Matches sequential baseline: "
          f"{matches_baseline(os.path.join(dataset_dir, BASELINE_CSV), total)}")
    print(f"\nTop {TOP_N} query words in the corpus:")
    for word, count in total.most_common(TOP_N):
        print(f"  {word}: {count}")
    print(f"\nEXECUTION_TIME={elapsed:.6f}")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    comm.Barrier()
    t0 = MPI.Wtime()

    query = load_query(os.path.join(dataset_dir, QUERY_NAME)) if rank == 0 else None
    targets = set(comm.bcast(query, root=0))

    if rank == 0:
        files = list_corpus(dataset_dir)
        chunks = lpt_partition(dataset_dir, files, size)
    else:
        files, chunks = None, None
    my_files = comm.scatter(chunks, root=0)

    t_local = MPI.Wtime()
    freq, tokens = count_files(dataset_dir, my_files, targets)
    t_local = MPI.Wtime() - t_local

    my_bytes = sum(os.path.getsize(os.path.join(dataset_dir, n)) for n in my_files)
    partials = comm.gather(
        {"rank": rank, "files": len(my_files), "bytes": my_bytes,
         "tokens": tokens, "time": t_local, "freq": freq},
        root=0,
    )

    if rank == 0:
        total = Counter()
        for p in partials:
            total.update(p["freq"])
        elapsed = MPI.Wtime() - t0
        report(partials, total, elapsed, len(files), size, dataset_dir,
               "MPI v2 (size-aware LPT distribution)")
        save_csv(os.path.join(dataset_dir, OUTPUT_CSV), total)


if __name__ == "__main__":
    main()
