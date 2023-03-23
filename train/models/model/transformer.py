"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder

from models.mask_utils import *


class Transformer(nn.Module):

    def __init__(self, cfg, vocab_size, SOS_IDX, PAD_IDX, device):
        super().__init__()
        self.src_pad_idx = PAD_IDX
        self.trg_pad_idx = PAD_IDX
        self.trg_sos_idx = SOS_IDX
        self.vocab_size = vocab_size
        self.device = device
        self.encoder = Encoder(d_model=cfg.get("d_model"),
                               n_head=cfg.get("n_head"),
                               max_len=cfg.get("max_len"),
                               ffn_hidden=cfg.get("ffn_hidden"),
                               enc_voc_size=vocab_size,
                               drop_prob=cfg.get("drop_prob"),
                               n_layers=cfg.get("n_layers"),
                               device=device)

        self.decoder = Decoder(d_model=cfg.get("d_model"),
                               n_head=cfg.get("n_head"),
                               max_len=cfg.get("max_len"),
                               ffn_hidden=cfg.get("ffn_hidden"),
                               dec_voc_size=vocab_size,
                               drop_prob=cfg.get("drop_prob"),
                               n_layers=cfg.get("n_layers"),
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx) * \
                   self.make_no_peak_mask(src, src)

        output = self.decoder(src, None, src_mask, None)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask

    def make_museformer_mask(self, q, k, q_bar_idx, k_bar_idx, struct_related_bars=[1,2,4,8,16,32], use_eos_as_summary=True):
        # Generate the mask for the seqeuence based on https://arxiv.org/pdf/2210.10349.pdf
        # Choice between calculating structure related bars based on similarity or using those suggested from the paper.

        # First split sequence into bars by the bar token
        q_bar_pointers = get_bar_locations(q, q_bar_idx)
        k_bar_pointers = get_bar_locations(k, k_bar_idx)

        """
        General masking scheme example where the 1st bar (x1-3) is a structure related bar for bar 3 (x6-8)

            | b1 x1 x2 x3 b2 x4 x5 b3 x6 x7 x8 b4
        Main mask
        x1  |    #
        x2  |    #  #
        x3  |    #  #  #  
        x4  |    #  #  #     #
        x5  |    #  #  #     #  #
        x6  |             #  #  #     #
        x7  |             #  #  #     #  #
        x8  |             #  #  #     #  #  #
        Summarisation Step
        b1  |
        b2  |    #  #  #  #
        b3  |                #  #  #
        b4  |                         #  #  #  #

        (first bar gets effectively ignored, and we would treat the EOS token as a end of bar if no final bar token exists)
        """
