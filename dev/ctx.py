from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch
import time

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to(torch.bfloat16).cuda()

prompt = "Once upon a time"

while True:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    start = time.perf_counter()
    print(tokenizer.decode(model.generate(input_ids, max_new_tokens=100)[0]))
    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")