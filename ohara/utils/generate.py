def generate(model, tokenizer, max_tokens=64, **kwargs):
    """
    Generate text from a model
    """
    model.eval()

    for _ in range(max_tokens):
        input_ids = torch.tensor(tokenizer.encode("")).unsqueeze(0)
        with torch.no_grad():
            outputs = model.generate(input_ids, **kwargs)
            tokenizer.decode(outputs[0].tolist())
