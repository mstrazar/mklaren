#!/bin/bash
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
for i in $(seq 1 9) ; do
    echo python.py energy periodic T$i
    python energy.py periodic T$i 1>T$i.out.txt 2>T$i.err.txt &
    sleep 1
done

echo "Waiting ..."
wait
echo "End"