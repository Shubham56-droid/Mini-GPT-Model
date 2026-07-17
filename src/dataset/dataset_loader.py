from pathlib import Path

class DatasetLoader:
    def __init__(self,dataset_name="dictionary"):
        self.dataset_name = dataset_name
    
    def load(self):
        if self.dataset_name == "dictionary":
            file_path = Path('data/raw/dictionary.txt')
        elif self.dataset_name == "tinystories":
            file_path = Path("data/tinystories/train.txt")
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}"
            )
        
        with open(file_path,"r",encoding="utf-8") as f:
            text = f.read()
        
        return text
