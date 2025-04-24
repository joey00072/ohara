
from dataclasses import dataclass
from collections import OrderedDict

import yaml
import json


class DefaultConfig(OrderedDict):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def load_from_yaml(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            yaml_config = yaml.load(f)
        for key, value in yaml_config.items():
            setattr(self, key, value)
            
    def save_to_yaml(self, yaml_path: str):
        with open(yaml_path, "w") as f:
            yaml.dump(self, f)

    def load_from_json(self, json_path: str):
        with open(json_path, "r") as f:
            json_config = json.load(f)
        for key, value in json_config.items():
            setattr(self, key, value)
            
    def save_to_json(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(self, f)
            
    def load_from_dict(self, config_dict: dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
            
    def save_to_dict(self):
        return self.__dict__
    
    @classmethod
    def load_from_hf(cls, model_name: str):
        url = f"https://huggingface.co/{model_name}/raw/main/config.json"
        import requests
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError(f"Failed to load config from {url}")
        d = r.json()    
        df = cls()
        for key, value in d.items():
            df[key] = value
        return df


        
        
if __name__ == "__main__":
    model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"
    config = DefaultConfig.load_from_hf(model_name)
    print(config)
    print(config.vocab_size)
