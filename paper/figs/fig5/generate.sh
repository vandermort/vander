#!/usr/bin/env bash

./generate_data.sh > data.csv

python plot.py --data data.csv --prop rad_p_1
python plot.py --data data.csv --prop rad_p_1 --slack-dims 16
