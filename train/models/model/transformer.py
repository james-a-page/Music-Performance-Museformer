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

from models.mask_utils import get_bar_locations


class Transformer(nn.Module):

    def __init__(self, cfg, pretrain, SOS_IDX, PAD_IDX, device):
        super().__init__()
        self.src_pad_idx = PAD_IDX
        self.trg_pad_idx = PAD_IDX
        self.trg_sos_idx = SOS_IDX
        self.device = device
        self.pretrain = pretrain
        self.attention_method = "museformer"

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
        #Find bar postionts before embedding
        tgt_bar_positions = [i for i,tok in enumerate(tgt[0]) if torch.equal(tok, torch.Tensor([3, 3, 3, 3, 3, 3, 3, 3]).to(self.device))]
        src = self.emb(src)
        tgt = self.emb(tgt)

        src_mask = self.make_pad_mask(src_pad_mask, src_pad_mask)
        src_trg_mask = self.make_pad_mask(tgt_pad_mask, src_pad_mask)

        if self.attention_method == "museformer":
            trg_mask = self.make_pad_mask(tgt_pad_mask, tgt_pad_mask) * \
                    self.make_museformer_mask(tgt_pad_mask, tgt_pad_mask, tgt_bar_positions)
        else:
            trg_mask = self.make_pad_mask(tgt_pad_mask, tgt_pad_mask) * \
                    self.make_no_peak_mask(tgt_pad_mask, tgt_pad_mask)

        enc_src = self.encoder(src, src_mask)
        if self.pretrain:
            output = self.decoder(tgt, None, trg_mask, src_trg_mask)
        else:
            output = self.decoder(tgt, enc_src, trg_mask, src_trg_mask)
# 
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
        mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )

        return mask


    """
    Generate the mask for the seqeuence based on https://arxiv.org/pdf/2210.10349.pdf

    General masking scheme example where the 1st bar (x1-3) is a structure related bar for bar 3 (x6-8)

        | b1 x1 x2 x3 b2 x4 x5 b3 x6 x7 x8 b4
    Main mask
    b1  |
    x1  |    #
    x2  |    #  #
    x3  |    #  #  #
    x4  |    #  #  #     #
    x5  |    #  #  #     #  #
    b2  |    #  #  #  #
    b3  |                #  #  #
    x6  |             #  #  #     #
    x7  |             #  #  #     #  #
    x8  |             #  #  #     #  #  #
    b4  |                         #  #  #  #


    """

    def make_museformer_mask(
        self,
        q,
        k,
        summary_positon,
        struct_related_bars=[1, 2, 4, 8, 16, 32],
        use_eos_as_summary=False,
        EOS_idx=3,
    ):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        tri_mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )
        museformer_mask = torch.zeros_like(tri_mask).to(self.device)

        bar_count = 0

        first_of_bar = False

        for i, mask in enumerate(tri_mask):
            temp_mask = torch.zeros_like(mask)
            if bar_count < len(summary_positon) and i == summary_positon[bar_count]:
                if bar_count == 0:
                    #First bar is all tokens from idx 1 -> summary_position[0]
                    temp_mask.index_fill_(0, torch.arange(1, i).to(self.device), 1)
                else:
                    #All other bars cover all tokens after the previous summary_positon idx to the current summary_positon idx
                    temp_mask.index_fill_(0, torch.arange(summary_positon[bar_count - 1] + 1, i).to(self.device), 1)
                first_of_bar = True
                bar_count += 1
            else:
                if bar_count > 0:
                    # unmask all but current bar and previous and then struct_related_bar tokens
                    if bar_count == 1:
                        prev_bar_start = 1
                    else:
                        prev_bar_start = summary_positon[bar_count - 2] + 1

                    if first_of_bar:
                        # The previous bar but remove the summary token for that bar.
                        # temp_mask.index_fill_(0, torch.arange(prev_bar_start + 1, i).to(self.device), 1)
                        # temp_mask[i-1] = False
                        for p, pos in enumerate(summary_positon):
                            if p > bar_count:
                                break
                            if bar_count - p in struct_related_bars:
                                temp_mask[pos] = True
                        # Struct_related_bar
                        for struct_bar in struct_related_bars:
                            if bar_count > struct_bar:
                                struct_bar_start = summary_positon[bar_count - struct_bar - 1] + 1
                                struct_bar_end = summary_positon[bar_count - struct_bar] - 1

                                temp_mask.index_fill_(0, torch.arange(struct_bar_start, struct_bar_end).to(self.device), 1)
                            else:
                                break
                        first_of_bar = False
                    else:
                        temp_mask = museformer_mask[i-1]
                        temp_mask[i] = True
                else:
                    # unmask all but current bar
                    if first_of_bar:    
                        temp_mask.index_fill_(0, torch.arange(1, i).to(self.device), 1)
                        temp_mask[i-1] = False
                        first_of_bar = False
                    else:
                        temp_mask = museformer_mask[i-1]
                        temp_mask[i] = True

            museformer_mask[i] = temp_mask
        return museformer_mask

        # for i, mask in enumerate(tri_mask):
        #     temp_mask = torch.zeros_like(mask)
        #     if bar_count < len(summary_positon) and i == summary_positon[bar_count]:
        #         if bar_count == 0:
        #             #First bar is all tokens from idx 1 -> summary_position[0]
        #             temp_mask.index_fill_(0, torch.arange(1, i).to(self.device), 1)
        #         else:
        #             #All other bars cover all tokens after the previous summary_positon idx to the current summary_positon idx
        #             temp_mask.index_fill_(0, torch.arange(summary_positon[bar_count - 1] + 1, i).to(self.device), 1)
        #         first_of_bar = True
        #         bar_count += 1
        #     else:
        #         if bar_count > 0:
        #             # unmask all but current bar and previous and then struct_related_bar tokens
        #             if bar_count == 1:
        #                 prev_bar_start = 1
        #             else:
        #                 prev_bar_start = summary_positon[bar_count - 2] + 1

        #             if first_of_bar:
        #                 # The previous bar but remove the summary token for that bar.
        #                 temp_mask.index_fill_(0, torch.arange(prev_bar_start + 1, i).to(self.device), 1)
        #                 temp_mask[i-1] = False
        #                 # Struct_related_bar
        #                 for struct_bar in struct_related_bars:
        #                     if bar_count > struct_bar:
        #                         temp_mask[summary_positon[bar_count - struct_bar]] = True
        #                     else:
        #                         break
        #                 first_of_bar = False
        #             else:
        #                 temp_mask = museformer_mask[i-1]
        #                 temp_mask[i] = True
        #         else:
        #             # unmask all but current bar
        #             if first_of_bar:    
        #                 temp_mask.index_fill_(0, torch.arange(1, i).to(self.device), 1)
        #                 temp_mask[i-1] = False
        #                 first_of_bar = False
        #             else:
        #                 temp_mask = museformer_mask[i-1]
        #                 temp_mask[i] = True

        #     museformer_mask[i] = temp_mask
        # return museformer_mask

    def _kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer._kl_divergence()
        return KLD
