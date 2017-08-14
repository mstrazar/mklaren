#!/usr/bin/env bash

# Run blitzer_sentiment for L2KRR; Features are still pre-filtered.
export PYTHONPATH=../..
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

for ddir in ../../datasets/rnacontext/full/* ; do
    for rank in 2 10 ; do
        dset=`basename $ddir`
        echo python rnacontext_regr.py $dset $rank
        python rnacontext_regr.py $dset $rank 2>$dset.2.err.txt 1>$dset.2.out.txt &
        sleep 1
    done
done

echo "Waiting..."
wait
echo "End."