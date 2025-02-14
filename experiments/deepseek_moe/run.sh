#! /bin/bash

echo "Baseline"
python train.py baseline
echo "--------------------------------"

echo "DeepSeek Moe + Aux-Loss"
python train.py use_aux_loss
echo "--------------------------------"

echo "DeepSeek Moe + aux_free (DSV3)"
python train.py aux_free_loadbalancing
echo "--------------------------------"









