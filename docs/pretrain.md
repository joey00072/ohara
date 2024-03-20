
first you need to download dataset and pretokenize it for fater training
<br>



## download and pretokenize
```python
dataset="roneneldan/TinyStories"
tokenizer="microsoft/phi-2"

print(f"Pretokenizing {dataset=} with {tokenizer=}")

preprocessor = DatasetPreprocessor(
    dataset_name=dataset,
    tokenizer_name=tokenizer,
    splits=["train", "validation"],
)

preprocessor.process_and_save()

```
In example I Do this for tinystoires,<br> 
just change dataset and tokenizer if you want it will download it from hugging face and start pretokenize it.<br>
Depending on dataset it will take a while.<br>
```bash
python examples/prepare-dataset.py
```

## train 
### script
This contain simple traning code,
Change the hyper paraqmeters in train.py to fit it on your gpu,
to `wandb login` for cli,

```bash
python examples/train_llama.py
```

### lighting fabric (recommanded)
recommanded because  <br>
This uses lighting fabric to train, fabric give me some quality extras whichout too much code changes,<br>
but also I plan to add more things in the future. eg multi-gpu traning, fp16 training, etc. (shoube easy bcs fabric)
```bash
python examples/train_llama_fabric.py 
```

You can check the training in wandb it something like this. this will take a while so grab a cup of coffee.

![train](./src/image.png)

