"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder
from models.embedding.octuple_embedding import OctupleEmbedding


class Transformer(nn.Module):

    def __init__(self, cfg, SOS_IDX, PAD_IDX, device):
        super().__init__()
        self.src_pad_idx = PAD_IDX
        self.trg_pad_idx = PAD_IDX
        self.trg_sos_idx = SOS_IDX
        self.device = device

        self.encoder = Encoder(d_model=cfg.get("d_model"),
                               n_head=cfg.get("n_head"),
                               max_len=cfg.get("max_len"),
                               ffn_hidden=cfg.get("ffn_hidden"),
                               drop_prob=cfg.get("drop_prob"),
                               n_layers=cfg.get("n_layers"),
                               bayes_compression=cfg.get("bayes_compression"),
                               device=device)

        self.decoder = Decoder(d_model=cfg.get("d_model"),
                               n_head=cfg.get("n_head"),
                               embedding_sizes=cfg["embedding_sizes"],
                               ffn_hidden=cfg.get("ffn_hidden"),
                               drop_prob=cfg.get("drop_prob"),
                               n_layers=cfg.get("n_layers"),
                               bayes_compression=cfg.get("bayes_compression"),
                               device=device)

        # Components including kl_divergence
        self.kl_list = [self.encoder, self.decoder]

        self.emb = OctupleEmbedding(embedding_sizes=cfg["embedding_sizes"],
                                    d_model=cfg.get("d_model"),
                                    max_len=cfg.get("max_len"),
                                    drop_prob=cfg.get("drop_prob"),
                                    PAD_IDX=PAD_IDX,
                                    device=device)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
        src = self.emb(src)
        tgt = self.emb(tgt)

        src_mask = self.make_pad_mask(src_pad_mask, src_pad_mask)
        src_trg_mask = self.make_pad_mask(tgt_pad_mask, src_pad_mask)
        

        trg_mask = self.make_pad_mask(tgt_pad_mask, tgt_pad_mask) * \
                   self.make_no_peak_mask(tgt_pad_mask, tgt_pad_mask)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask

    def _kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer._kl_divergence()
        return KLD