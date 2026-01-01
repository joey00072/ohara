# Ohara

This is my collection of implimention of llm,paper and things I hand in my mind
I hand lot of fragmented code of implimention of diffrent model 

this is attempt to make it eveything in one place <br>
This lib is for runing/copying code for expriments
<br>
install with
```bash
git clone https://github.com/joey00072/ohara.git
pip install -e .
```
then

```bash
## download and pretokenize
python examples/prepare-dataset.py

## train 
## look at train.py its fairly easy
python examples/train_llama.py

# lighting fabric verison is also avalible (recommanded)
python examples/train_llama_fabric.py 
```
![alt text](./docs/src/image.png)

llama-20M trained on tinystores for 1.7B

inferance on phi2
```zsh
## this will download model from hf and run it in torch.flaot16

python phi_inference.py 

## look at files and you can impliment rest of things easily, 
## I belive in you 😉
```


###  The lib to maximize FAFO
papaers and theory is on one side but `code is truth`, in the end things that matter that works (runs)<br>
If you look into [docs](./docs/) you can find some written things. this are mostly copied from my obsidian notes


### WORK IS PROGESS (always)

### papers / models
- [TokenFormer](./experiments/tokenformer/pattention.py)
- [MLA](./experiments/mla/mla.py)
- [Griffin & Hawk](./experiments/griffin_and_hawk/griffin_and_hawk.py)
- [Galore](./experiments/galore/galore.py)
- [Qsparse](./experiments/qsparse/qsparse.py)
- [Bitnet](./experiments/bitnet/bitnet.py)
- [renet](./ohara/models/retnet.py)
- [Alibi Embeddings](./ohara/embeddings_pos/alibi.py) | [md](./ohara/embeddings_pos/alibi/alibi.md)
- [Rotary Embeddings](./ohara/embeddings_pos/rotatry.py) | [md](./docs/RoFormer.md) 
- [LoRA ](./ohara/adaptor/lora.py)
- [LLAMA](./ohara/llama/llama.py) | [md](./docs/llama/llama.md)
- [XPOS](./ohara/embeddings_pos/xpos.py)
- [Mamba](./ohara/models/mamba.py)
- [GPT](./ohara/models/gpt.py) | [md](./docs/gpt/gpt.md)


### More things are not in this repo
1. [TinyLora](https://github.com/joey00072/TinyLora)
2. [Neural Style Transfer in Pytorch](https://github.com/joey00072/Neural-Style-Transfer-in-Pytorch)



## TODO  (A lot)
- [ ] make infercer class better
- [ ] make training loop better (use lightning fabric maybe)
- [ ] Finetuning in structed way (I just rawdoag code when I need it)
- [ ] DPO 
- [ ] make is py modele so I can create expriment folder and put all this in it

## Fund My Caffeine Addiction 
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R6R8KQTZ5)


### contribution guidelines
- be nice, 
- code explaintions || docs are appricated
- memes on pr recommend

