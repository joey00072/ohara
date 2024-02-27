from ohara.pretokenize import DatasetPreprocessor,OpenHermesDatasetPreprocessor



def prepare_openhermes_2_5():
    preprocessor = OpenHermesDatasetPreprocessor(
        dataset_name="teknium/OpenHermes-2.5",
        tokenizer_name="philschmid/gemma-tokenizer-chatml",
        revision="f7e624a58ce3642ec50483cb6039468ee8c3c464",
        splits=["train"],
        num_proc=1,

        )
    
    preprocessor.process_and_save(remove_columns=None)
    

if __name__ == "__main__":
    tokenizer_name: str = "microsoft/phi-2"
    dataset_name = "roneneldan/TinyStories"
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset_name, tokenizer_name=tokenizer_name, splits=['train', 'validation']
    )
    preprocessor.process_and_save()
