
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, device):
        self.device = device

        dic = torch.load(path)
        self.data = dic[0].type(torch.float32)
        self.label = dic[1].type(torch.int64)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx].to(self.device).unsqueeze(0), \
                self.label[idx].to(self.device).unsqueeze(0)
