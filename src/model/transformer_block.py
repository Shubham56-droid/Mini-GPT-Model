import torch
import torch.nn as nn

from src.model.multi_head_attention import MultiHeadAttention
from src.model.feed_forward import FeedForward
from src.model.layer_norm import LayerNormalization

class TransformerBlock(nn.Module):

    def __init__(self,embedding_dim:int,num_heads:int,hidden_dim:int):
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim=embedding_dim,num_heads=num_heads)

        self.norm1 = LayerNormalization(embedding_dim)

        self.ffn = FeedForward(embedding_dim=embedding_dim,hidden_dim=hidden_dim)

        self.norm2 = LayerNormalization(embedding_dim)


    def forward(self,x):

        Q,K,V,scores,mask,attention_weights,context,attention_output = self.attention(x)

        residual_1 = x + attention_output

        norm1 = self.norm1(residual_1)

        hidden,activated,ffn_output = self.ffn(norm1)

        residual_2 = norm1 + ffn_output

        output = self.norm2(residual_2)

        return (
            Q,
            K,
            V,
            scores,
            mask,
            attention_weights,
            context,
            hidden,
            activated,
            output,
        )