#!/usr/bin/env bash

K=5
PART=1
for V in 1000 5000 10000
do
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*48*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --header
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*32*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*16*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --footer
	echo -e '\n\n'
done

K=10
PART=1
for V in 1000 5000 10000
do
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*64*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --header
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*48*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*32*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --footer
	echo -e '\n\n'
done

K=5
PART=123
for V in 1000 5000 10000
do
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*48*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --header
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*32*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*16*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --footer
	echo -e '\n\n'
done

K=10
PART=123
for V in 1000 5000 10000
do
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*64*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --header
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*48*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p
	vndr.generate.test_results --results ../../experiments/bioasq/experiments/*-[248]k-part-$PART-k-$K-*$V-*32*/analysis/test-metrics.json --attributes model ndcg@5 macrof1 prec@5 "exact acc" argmax_p e-argmax_p --footer
	echo -e '\n\n'
done
