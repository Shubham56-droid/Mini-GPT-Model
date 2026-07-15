import math
import torch
import torch.nn as nn 

class PositionalEncoding(nn.Module):

    def __init__(self,embedding_dim:int,max_length:int = 5000):
        super().__init__()
        # Create matrix of shape max_legth, embedding_dim 
        pe = torch.zeros(max_length,embedding_dim)

        # Positional column
        position = torch.arange(0,max_length).unsqueeze(1).float()

        # Calculate frequiecies 
        div_term = torch.exp(
            torch.arange(0,embedding_dim,2).float() * (-math.log(10000.0)/embedding_dim)
        )

        # Apply sine to even column
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd column
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Save as non-trainable tensor 
        self.register_buffer("pe",pe)

    def forward(self,x):
    # x shape
    # (batch_size, sequence_length, embedding_dim)

        seq_length = x.size(1)

        return x + self.pe[:,:seq_length,:]



