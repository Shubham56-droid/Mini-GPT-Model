import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self,embedding_dim:int, hidden_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim,hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim,embedding_dim)

    def forward(self,x):
        hidden = self.fc1(x)
        activated = self.activation(hidden)
        output = self.fc2(activated)
        return hidden,activated,output
