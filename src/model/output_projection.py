import torch.nn as nn

class OutputProjection(nn.Module):

    def __init__(self,embedding_dim:int,vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(
            embedding_dim,
            vocab_size
        )
    
    def forward(self,x):
        logits = self.linear(x)
        return logits