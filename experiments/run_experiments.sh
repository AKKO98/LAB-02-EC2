#!/bin/sh
# Full experimental plan, run inside augustosalazar/slim-mpi:2.
# The code is copied to /work (container filesystem) so timings are not affected
# by the Windows/macOS bind-mount layer; logs are written back to /app/experiments/results.

set -eu

SRC=/app
WORK=/work
OUT="${OUT:-$SRC/experiments/results}"
LOGS="$OUT/logs"
RUNS="${RUNS:-5}"
NP="${NP:-}"
MPIRUN="mpirun --allow-run-as-root --oversubscribe"

rm -rf "$WORK" "$OUT"
mkdir -p "$WORK" "$LOGS"
cp "$SRC/generator.py" "$SRC/baseline_secuencial.py" "$SRC/mpi1.py" "$SRC/mpi2.py" \
   "$SRC/detectar_procesos_mpi.py" "$WORK/"
cd "$WORK"

now() { python -c 'import time; print(f"{time.time():.6f}")'; }

echo "== Environment" | tee "$LOGS/environment.log"
{
  echo "nproc=$(nproc)"
  grep -m1 "model name" /proc/cpuinfo
  grep MemTotal /proc/meminfo
  python --version
  python -c "import mpi4py; print('mpi4py', mpi4py.__version__)"
  mpirun --version | head -n 1
} | tee -a "$LOGS/environment.log"

echo "== Generating dataset"
python generator.py > "$LOGS/generator.log"
tail -n 8 "$LOGS/generator.log"

if [ -z "$NP" ]; then
  echo "== Detecting process count"
  python detectar_procesos_mpi.py | tee "$LOGS/detect.log"
  NP=$(grep "N RECOMENDADO" "$LOGS/detect.log" | awk '{print $3}')
fi
echo "Using p=$NP"
echo "$NP" > "$OUT/np.txt"

echo "run,program,np,wall_seconds" > "$OUT/wall_times.csv"
for RUN in $(seq 1 "$RUNS"); do
  echo "== Run $RUN/$RUNS"

  T0=$(now)
  python baseline_secuencial.py > "$LOGS/baseline_run${RUN}.log"
  T1=$(now)
  echo "$RUN,baseline,1,$(python -c "print(f'{$T1-$T0:.6f}')")" >> "$OUT/wall_times.csv"
  grep "Tiempo" "$LOGS/baseline_run${RUN}.log"

  for PROG in mpi1 mpi2; do
    T0=$(now)
    $MPIRUN -np "$NP" python "$PROG.py" > "$LOGS/${PROG}_np${NP}_run${RUN}.log"
    T1=$(now)
    echo "$RUN,$PROG,$NP,$(python -c "print(f'{$T1-$T0:.6f}')")" >> "$OUT/wall_times.csv"
    grep -E "EXECUTION_TIME|LI |Matches" "$LOGS/${PROG}_np${NP}_run${RUN}.log"
  done
done

python "$SRC/experiments/summarize.py" "$OUT"
