#!/usr/bin/env bash

N_SAMPLES=20000

N_LINES=2
for N in 50 100 200 500 1000 2000 5000 10000
do
	for D in 3 5 7 9 11
	do
		OUT=$(python check_infeasible_cyclic.py --N $N --D $D --num-samples $N_SAMPLES --param gale)
		echo "$OUT" | tail -n $N_LINES
		N_LINES=1
	done
done

for N in 50 100 200 500 1000 2000 5000 10000
do
	for D in 3 5 7 9 11
	do
		OUT=$(python check_infeasible_cyclic.py --N $N --D $D --num-samples $N_SAMPLES --param gale --slack-dims 16)
		echo "$OUT" | tail -n $N_LINES
		N_LINES=1
	done
done
