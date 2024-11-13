#!/bin/bash
echo "\n<GLU>\n\n\n"
python train.py  --mlp=GLU --expand_ratio=4
echo "\n</GLU>\n\n\n"
echo "\n<XGLU>\n\n\n"
python train.py  --mlp=XGLU --expand_ratio=4
echo "\n</XGLU>\n\n\n"
echo "\n<RGLU>\n\n\n"
python train.py  --mlp=RGLU --expand_ratio=8
echo "\n</RGLU>\n\n\n"
