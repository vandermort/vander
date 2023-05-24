#!/bin/bash
echo -e "Downloading data splits..."
echo -e "Saving files to data folder..."
mkdir -p data
wget https://vander-experiments.s3.eu-west-2.amazonaws.com/experiments.tar.gz -O - | tar -xz
