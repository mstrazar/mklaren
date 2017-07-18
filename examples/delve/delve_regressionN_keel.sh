#!/usr/bin/env bash

# Run blitzer_sentiment for L2KRR; Features are still pre-filtered.
export PYTHONPATH=../..
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

for ddir in ailerons pole elevators california house mv ; do
    dset=`basename $ddir`
    echo python delve_regressionN.py $dset
    python delve_regressionN.py $dset 2>$dset.N.err.txt 1>$dset.N.out.txt &
    sleep 1
done

echo "Waiting..."
wait
echo "End."