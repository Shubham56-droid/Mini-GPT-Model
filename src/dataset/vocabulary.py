import json 

class Vocabulary:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
    
    def build(self,tokens:list[str]):
        unique_words = sorted(set(tokens))
        
        for index,word in enumerate(unique_words):
            self.word_to_id[word] = index
            self.id_to_word[index] = word
        
    def encode(self,tokens:list[str]) -> list[int]:
        return [self.word_to_id[word] for word in tokens]
    
    def decode(self,ids:list[int])->list[str]:
        return [self.id_to_word[idx] for idx in ids]
    
    def size(self):
        return len(self.word_to_id)
    
    def save(self,filepath:str):

        with open(filepath,"w",encoding="utf-8") as file:
            json.dump(self.word_to_id,file,indent=4)
    
    def load(self,filepath:str):
        
        with open(filepath,"r",encoding="utf-8") as file:
            self.word_to_id = json.load(file)
        
        self.id_to_word = {
            int(idx): word
            for word,idx in self.word_to_id.items()
        }