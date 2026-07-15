import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self,embedding_dim:int,num_heads:int):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        #Learnable matrices
        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)

        #final projection
        self.output = nn.Linear(embedding_dim,embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):

        batch_size, seq_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)


        # Split into multiple heads

        Q = Q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K = K.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V = V.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        # Shape
        # (heads, sequence_length, head_dim)

        # Attention Score
        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores = scores / math.sqrt(self.head_dim)

        # Shape

        # (heads, sequence_length, sequence_length)

       #----------------------------------
       # CASUAL MASK
       #----------------------------------
        mask = torch.tril(torch.ones(seq_length,seq_length,device=x.device))

        mask = mask.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(mask==0,float("-inf"))
       #---------------------------------

        # SOFTMAX
        attention = self.softmax(scores)

        # CONTEXT 
        context = torch.matmul(attention,V)

        # Shape
        # (heads, sequence_length, head_dim)

        #Put sequence first again
        context = context.transpose(1,2)

        # Shape
        # (sequence_length, heads, head_dim)

        # merge all the heads
        context = context.contiguous().view(
            batch_size,
            seq_length,
            self.embedding_dim
        )

        # Shape
        # (sequence_length, embedding_dim)

        #Final Linear Layer
        output = self.output(context)

        return (
            Q,
            K,
            V,
            scores,
            mask,
            attention,
            context,
            output
        )
    

    
