#!/usr/bin/env bash

# Run blitzer_sentiment for L2KRR; Features are still pre-filtered.
export PYTHONPATH=../..
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5
export OMP_NUM_THREADS=5

for dset in boston abalone comp bank pumadyn kin ionosphere census ; do
    echo python delve_regression2.py $dset
    python delve_regression2.py $dset 2>$dset.2.err.txt 1>$dset.2.out.txt &
    sleep 1
done

echo "Waiting..."
wait
echo "End."