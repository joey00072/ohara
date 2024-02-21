import pygame
from pygame.joystick import Joystick

# from rich import print

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() < 1:
    print(f"Joeystick not found: {pygame.joystick.get_count()}")
    exit(0)


stick = Joystick(0)

print(stick.get_name())


from lightning.data import optimize
from functools import partial
import pyarrow.parquet as pq
from pathlib import Path


# 1. Function to tokenize the text contained within the Slimpajam files
def tokenize_fn(filepath, tokenizer=None):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
        for text in batch.to_pandas()["content"]:
            yield tokenizer.encode(text, bos=False, eos=True)


# 2. Generate the inputs (we are going to optimize all the compressed json files from SlimPajama dataset )
input_dir = (
    "/teamspace/s3_connections/tinyllama-template/starcoderdata"
)  # "/teamspace/studios/StarCoder_Dataset/data"
inputs = [str(file) for file in Path(input_dir).rglob("*.parquet")]

# 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
outputs = optimize(
    fn=partial(
        tokenize_fn, tokenizer=Tokenizer("./checkpoints/Llama-2-7b-hf")
    ),  # Note: You can use HF tokenizer or any others
    inputs=inputs,
    output_dir="/teamspace/datasets/starcoder/",
    chunk_size=(2049 * 8012),
    num_nodes=16,
    machine=Machine.DATA_PREP,  # use 32 CPU machine
    reorder_files=True,  # split evenly the files based on their size across all machines
)
