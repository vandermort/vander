#!/usr/bin/env bash
mkdir -p plots

for K in 5 10
do
	for PART in 1 123
	do
		vndr.plot_feasible --results ../../../experiments/bioasq/experiments/sigmoid*-$PART-k-$K-*/analysis/test-metrics.json --savefig plots/csl-$PART-k-$K.eps
		vndr.plot_feasible --results ../../../experiments/bioasq/experiments/vander-[248]k*-$PART-k-$K-*/analysis/test-metrics.json --savefig plots/vander-dft-l-$PART-k-$K.eps
		vndr.plot_feasible --results ../../../experiments/bioasq/experiments/vander-fft*-$PART-k-$K-*/analysis/test-metrics.json --savefig plots/vander-dft-$PART-k-$K.eps
	done
done
