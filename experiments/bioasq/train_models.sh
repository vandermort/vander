#!/usr/bin/env bash

MAX_ITERS=50000
EVAL_EVERY=500
BATCH_SIZE=1024
LR=1.e-3

echo -e "####################################################################"
echo -e "####################   VOCAB 1000   ################################"
echo -e "####################################################################"

mkdir -p logs

VOCAB_DIM=1000
DATA_SIZE=2k
for K in 5 10
do
	for Ds in 16 32 48
	do
		for PART in 1 123
		do
			D=$(python -c "print($Ds if $K == 5 else $Ds + 16)")
			VANDER_D=$(expr $K \* 2 + 1)
			SLACK_D=$(expr $D - $VANDER_D)
			# Below is dimensionality of sigmoid features such that number of
			# parameters is comparable to vander
			SIG_D=$(vndr.compute_params --V $VOCAB_DIM --S $SLACK_D --K $K)
			vndr.train --blueprint blueprints/sigmoid-bottleneck.yaml --paths.experiment_name sigmoid-$SIG_D-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.feature_dim $SIG_D  --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
			vndr.train --blueprint blueprints/vander-fft.yaml --paths.experiment_name vander-fft-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.k $K --output_layer.slack_dims $SLACK_D  --output_layer.use_init true --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
			vndr.train --blueprint blueprints/vander.yaml --paths.experiment_name vander-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.k $K --output_layer.slack_dims $SLACK_D  --output_layer.use_init true --output_layer.freeze false --output_layer.param gale --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
		done
	done
done

echo -e "####################################################################"
echo -e "####################   VOCAB 5000   ################################"
echo -e "####################################################################"
VOCAB_DIM=5000
DATA_SIZE=4k
for K in 5 10
do
	for Ds in 16 32 48
	do
		for PART in 1 123
		do
			D=$(python -c "print($Ds if $K == 5 else $Ds + 16)")
			VANDER_D=$(expr $K \* 2 + 1)
			SLACK_D=$(expr $D - $VANDER_D)
			# Below is dimensionality of sigmoid features such that number of
			# parameters is comparable to vander
			SIG_D=$(vndr.compute_params --V $VOCAB_DIM --S $SLACK_D --K $K)
			vndr.train --blueprint blueprints/sigmoid-bottleneck.yaml --paths.experiment_name sigmoid-$SIG_D-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.feature_dim $SIG_D  --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
			vndr.train --blueprint blueprints/vander-fft.yaml --paths.experiment_name vander-fft-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.k $K --output_layer.slack_dims $SLACK_D  --output_layer.use_init true --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
			vndr.train --blueprint blueprints/vander.yaml --paths.experiment_name vander-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.k $K --output_layer.slack_dims $SLACK_D  --output_layer.use_init true --output_layer.freeze false --output_layer.param gale --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
		done
	done
done

echo -e "####################################################################"
echo -e "####################   VOCAB 10000   ################################"
echo -e "####################################################################"
VOCAB_DIM=10000
DATA_SIZE=8k
for K in 5 10
do
	for Ds in 16 32 48
	do
		for PART in 1 123
		do
			D=$(python -c "print($Ds if $K == 5 else $Ds + 16)")
			VANDER_D=$(expr $K \* 2 + 1)
			SLACK_D=$(expr $D - $VANDER_D)
			# Below is dimensionality of sigmoid features such that number of
			# parameters is comparable to vander
			SIG_D=$(vndr.compute_params --V $VOCAB_DIM --S $SLACK_D --K $K)
			vndr.train --blueprint blueprints/sigmoid-bottleneck.yaml --paths.experiment_name sigmoid-$SIG_D-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.feature_dim $SIG_D  --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
			vndr.train --blueprint blueprints/vander-fft.yaml --paths.experiment_name vander-fft-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.k $K --output_layer.slack_dims $SLACK_D  --output_layer.use_init true --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
			vndr.train --blueprint blueprints/vander.yaml --paths.experiment_name vander-$DATA_SIZE-part-$PART-k-$K-v-$VOCAB_DIM-d-$D --data.train_path "data/subsets-k-$K-v-$VOCAB_DIM/train-$DATA_SIZE-part-[$PART].json" --data.valid_path data/subsets-k-$K-v-$VOCAB_DIM/valid-$DATA_SIZE.json --data.labels_file data/subsets-k-$K-v-$VOCAB_DIM/vocab.txt --output_layer.out_dim $VOCAB_DIM --output_layer.k $K --output_layer.slack_dims $SLACK_D  --output_layer.use_init true --output_layer.freeze false --output_layer.param gale --max_iters 50000 --eval_every 500 --seed 32  --num_proc 1 --lr $LR --batch_size $BATCH_SIZE
		done
	done
done
