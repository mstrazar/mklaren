#!/bin/bash
export PYTHONPATH=../..
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
for i in $(seq 1 9) ; do
    echo python.py energy matern T$i
    python energy.py matern T$i 1>T$i.2.out.txt 2>T$i.2.err.txt &
    sleep 1
done

echo "Waiting ..."
wait
echo "End"