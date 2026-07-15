import torch
import random

class Dataloader:

    def __init__(self,tokens,context_length=8,batch_size=4,shuffle=True):
        self.tokens = tokens
        self.context_length = context_length
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.tokens) - self.context_length
    
    def __getitem__(self,index):
        input_tokens  = self.tokens[index:index+self.context_length]
    
        target_token = self.tokens[index+1:index+self.context_length+1]

        return input_tokens,target_token
    
    def get_batch(self,start_index):

        inputs = []
        targets = [] 

        for i in range(self.batch_size):

            input_tokens,target_tokens = self[start_index+i]
            inputs.append(input_tokens)
            targets.append(target_tokens)

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        return inputs,targets
    
    def get_epoch_indices(self):
        indices = list(range(len(self)))

        if self.shuffle:
            random.shuffle(indices)
            
        return indices

