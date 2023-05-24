# Vandermonde Output Layers for Provably Argmaxable k-Sparse Multi-Label Classification

This is the code to reproduce results in the paper.


# Installation

## Install python dependencies
```py
python3.8 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
# This installs all vndr.* commands
pip install -e .
```

## Set environment variables
```bash
# Avoid numpy hogging all threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Can also use cuda:0 etc.
export MLBL_DEVICE=cpu
# Update below to larger number for verification
export MLBL_NUM_PROC=1
```

## Run tests
```py
py.test
```

# Reproduce Results

## Train models from scratch
```bash
cd experiments/bioasq
./get_data.sh
# Set the root folder for the blueprints
sed -i -e "s|SETME|$PWD|g" blueprints/*.yaml
# Train all models from scratch
./train_models.sh
# Evaluate them on the test set
./eval.sh
```

# Explore Trained Models
All steps below assume:
*  The data has been downloaded, see above.
*  The experiments have been downloaded, see below.

## Download models
```bash
cd experiments/bioasq
./get_experiments.sh
sed -i -e "s|SETME|$PWD|g" experiments/*/blueprint.yaml
```

## Generate Figure 6
```bash
cd paper/figs/fig6
./generate.sh
```

## Generate Table 1 + Appendix Tables
```bash
cd paper/tables
./generate.sh
```

## Visualise results for models

```py
vndr.compare.plot --results experiments/sigmoid*/stats.tsv --attributes valid.loss valid.p5
```
