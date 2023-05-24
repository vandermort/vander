#!/bin/bash
echo -e "Downloading data splits..."
echo -e "Saving files to data folder..."
mkdir -p data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/data/subsets-k-5-v-1000.tar.gz -O - | tar -xz -C data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/data/subsets-k-5-v-5000.tar.gz -O - | tar -xz -C data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/data/subsets-k-5-v-10000.tar.gz -O - | tar -xz -C data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/data/subsets-k-10-v-1000.tar.gz -O - | tar -xz -C data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/data/subsets-k-10-v-5000.tar.gz -O - | tar -xz -C data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/data/subsets-k-10-v-10000.tar.gz -O - | tar -xz -C data
