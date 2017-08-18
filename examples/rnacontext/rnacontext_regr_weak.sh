#!/usr/bin/env bash

# Run blitzer_sentiment for L2KRR; Features are still pre-filtered.
export PYTHONPATH=../..
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

for rank in 10 ; do
    for ddir in ../../datasets/rnacontext/weak/*.txt.gz ; do
        dset=`basename $ddir`
        echo python rnacontext_regr.py $dset $rank
        python rnacontext_regr.py $dset $rank 2>$dset.$rank.2.err.txt 1>$dset.$rank.2.out.txt &
        sleep 1
    done
done

echo "Waiting..."
wait
echo "  End."