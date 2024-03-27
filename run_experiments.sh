#!/bin/bash

pip install Cython
pip install madmom
for i in {1..22}; do
    python train.py --experiment_id $i --encoding decoupled_dot --batch_size 16 --epochs 300 --patience 300
done
