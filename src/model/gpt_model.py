import torch.nn as nn 
from src.model.emdedding import WordEmbedding
from src.model.positional_encoding import PositionalEncoding
from src.model.transformer_block import TransformerBlock
from src.model.output_projection import OutputProjection


class GPTModel(nn.Module):

    def __init__(self,embedding_dim:int,hidden_dim:int,num_heads:int,num_layers:int,vocab_size: int):

        super().__init__()

        self.embedding = WordEmbedding(vocab_size=vocab_size,embedding_dim=embedding_dim)

        self.position = PositionalEncoding(embedding_dim=embedding_dim,max_length=5000)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        self.output_projection = OutputProjection(embedding_dim=embedding_dim,vocab_size=vocab_size)
    
    def forward(self, tokens, debug=False):
        x = self.embedding(tokens)
        x = self.position(x)

        debug_output = []

        for i,layer in enumerate(self.layers):
            Q, K, V, scores, mask, attention, context, hidden, activated, output = layer(x)

            x = output

            if debug:
                debug_output.append({
                    "layer": i + 1,
                    "Q": Q,
                    "K": K,
                    "V": V,
                    "scores": scores,
                    "mask": mask,
                    "attention": attention,
                    "context": context,
                    "hidden": hidden,
                    "activated": activated,
                    "output": output,
                })

        if debug:
            return debug_output
         

        logits = self.output_projection(x)
        return logits