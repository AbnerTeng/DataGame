#!/bin/bash

read -p "
Select your type:
1. train
2. eval

type number (1/2): " type

if [ $type == 1 ]; then
    python -m src.LM_script \
        --mode train \
        --n_epoch 5 \
        --lr 0.0001 \
        --exec mps
elif [ $type == 2 ]; then
    python -m src.LM_script \
        --mode eval \
        --exec mps
else
    echo "Wrong type number"
fi