#!/bin/bash

echo -e "Running partial KV experiment"

echo -e "---- Baseline ----"
python train.py attention_type=attention head_dim=32 num_heads=16

echo -e "---- K == V ----"
python train.py attention_type=k_is_v head_dim=32 num_heads=21

echo -e "---- Partial KV ----"
python train.py attention_type=partial_kv head_dim=32 num_heads=18
