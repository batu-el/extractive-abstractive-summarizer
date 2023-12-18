import torch

# DATASET CLASS: pytorch dataset class' subclass
class CNNDMdataset(torch.utils.data.Dataset):
    def __init__(self, X, y) -> None:
        self.source = X
        self.target = y
    def __getitem__(self, idx) -> torch.Tensor:
        # load one sample by index, e.g like this:
        source_sample = self.source[idx]
        target_sample = self.target[idx]
        # do some preprocessing, convert to tensor and what not
        return source_sample, target_sample
    def __len__(self):
        return len(self.source)