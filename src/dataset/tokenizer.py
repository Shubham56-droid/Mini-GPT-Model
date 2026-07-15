import re

class Tokenizer:
    def tokenize(self,text:str) -> list[str]:
        text = text.lower()

        text = re.sub(r"[^\w\s]","",text)

        tokens = text.split()
        return tokens
