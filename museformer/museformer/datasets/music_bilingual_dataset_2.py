import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class MusicBilingualDataset(BaseWrapperDataset):
    def __init__(self, datasets):
        (dataset1, dataset2) = datasets
        super().__init__(dataset1)
        self.dataset1 = dataset1
        self.dataset2 = dataset2


    def __getitem__(self, index):
        sample_src = self.dataset1[index]  # (len,)
        sample_tgt = self.dataset2[index]  # (len,)

        return torch.cat((sample_src[-1:], sample_src[:-1]), dim=0), sample_tgt

    def collater(self, samples):
        raise NotImplementedError
