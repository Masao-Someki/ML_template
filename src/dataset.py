# Script for dataset

import torch



class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_dir, device=None):
        dic = torch.load(train_dir)
        self.data = dic[0]
        self.label = dic[1]
        self.device = device

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return {'inputs': self.data[idx].to(self.device), 'labels': self.label[idx].to(self.device)}

