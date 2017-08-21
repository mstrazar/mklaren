#!/usr/bin/env bash

export PYTHONPATH=../..
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2

for ddir in ../../datasets/keel/* ; do
    dset=`basename $ddir`
    echo python distances_mkl_cv.py $dset
    python distances_mkl_cv.py $dset 2>$dset.cv.err.txt 1>$dset.cv.out.txt &
    sleep 1
done

echo "Waiting..."
wait
echo "End."