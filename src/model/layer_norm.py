import torch
import torch.nn as nn 

class LayerNormalization(nn.Module):
    
    def __init__(self,embedding_dim:int):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self,x):
        output = self.layer_norm(x)
        return output