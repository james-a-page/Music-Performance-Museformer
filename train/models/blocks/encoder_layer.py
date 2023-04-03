"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from models.layers.fast_attention import CausalSelfAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, bayes_compression):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.fast_self_attention = CausalSelfAttention(num_heads=n_head,
                                                       embed_dimension=d_model,
                                                       bias=False,
                                                       is_causal=False,
                                                       training=True,
                                                       bayes_compression=bayes_compression)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob, bayes_compression=bayes_compression)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # layers including kl_divergence
        self.kl_list = [self.ffn, self.fast_self_attention]

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x

        attention = "flash"
        if attention == "normal":
            x = self.attention(q=x, k=x, v=x)
        else:
            x = self.fast_self_attention(x=x)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

    def _kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer._kl_divergence()
        return KLD
