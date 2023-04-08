"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, bayes_compression, device):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  bayes_compression=bayes_compression)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        for layer in self.layers:
            x = layer(x, s_mask)

        return x

    def _kl_divergence(self):
        KLD = 0
        for layer in self.layers:
            KLD += layer._kl_divergence()
        return KLD
