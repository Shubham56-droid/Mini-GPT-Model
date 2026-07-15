# self attention layer 

import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self,embedding_dim:int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)

        self.softmax = nn.Softmax(dim=-1)
    
    # def forward(self, x):
    #     Q = self.query(x)
    #     K = self.key(x)
    #     V = self.value(x)

    #     scores = torch.matmul(Q, K.transpose(-2, -1))
        
    #     scores = scores / math.sqrt(self.embedding_dim)

    #     attention = self.softmax(scores)

    #     output = torch.matmul(attention, V)

    #     return output

    def forward(self,x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q,K.transpose(-2,-1))
        scores = scores / math.sqrt(self.embedding_dim)
        attention = self.softmax(scores)
        context = torch.matmul(attention,V)
        return Q, K, V, scores, attention, context



