echo "====================================================="
echo " Running: Residual Baseline "
echo "====================================================="
SEED=1337 CONNECTION_TYPE="residual" EXPANSION_RATE=1 MULTIPLE_OF=4.135 MAX_ITERS=5000 SAVE_CKPT_ITERS=0 PRINT_EVERY=1 python experiments/mHC/train.py
echo
echo "====================================================="
echo " Running: Manifold HyperConnections (mHC)"
echo "====================================================="
SEED=1337 CONNECTION_TYPE="mhc" EXPANSION_RATE=4 MULTIPLE_OF=4 MAX_ITERS=5000 SAVE_CKPT_ITERS=0 PRINT_EVERY=1 python experiments/mHC/train.py
echo
echo "====================================================="
echo " All runs completed! "
echo "====================================================="
