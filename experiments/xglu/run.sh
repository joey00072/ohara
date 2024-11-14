#!/bin/bash

# Run GLU experiment
echo -e "\nRunning GLU Baseline"
echo -e "=====================\n" 
python train.py --mlp=GLU --expand_ratio=4
echo -e "\nFinished GLU Baseline"
echo -e "=====================\n"

# Run XGLU experiment  
echo -e "\nRunning XGLU experiment"
echo -e "======================\n"
python train.py --mlp=XGLU --expand_ratio=4
echo -e "\nFinished XGLU experiment"
echo -e "======================\n"
