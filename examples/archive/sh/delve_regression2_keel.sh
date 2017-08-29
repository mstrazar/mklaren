#!/usr/bin/env bash

# Run blitzer_sentiment for L2KRR; Features are still pre-filtered.
export PYTHONPATH=../..
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

for ddir in ../../datasets/keel/* ; do
    dset=`basename $ddir`
    echo python delve_regression2.py keel $dset
    python delve_regression2.py keel $dset 2>$dset.2.err.txt 1>$dset.2.out.txt &
    sleep 1
done

echo "Waiting..."
wait
echo "End."