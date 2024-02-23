from ohara.pretokenize import DatasetPreprocessor


if __name__ == "__main__":
    tokenizer_name: str = "NeelNanda/gpt-neox-tokenizer-digits"
    dataset_name = "JeanKaddour/minipile"
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset_name, tokenizer_name=tokenizer_name, splits=["test"]
    )
    preprocessor.process_and_save()
