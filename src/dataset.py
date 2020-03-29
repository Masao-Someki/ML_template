
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, device):
        self.device = device

        dic = torch.load(path, map_location=torch.device('cpu'))
        self.data = dic[0]
        self.label = dic[1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'inputs': self.data[idx].to(device), 'labels': self.label[idx].to(device)}


