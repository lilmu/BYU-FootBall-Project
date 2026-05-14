import torch
from torch.utils.data import Dataset


class football_dataset(Dataset):
    def __init__(self, player_file, label_file):
        self.labels = torch.load(label_file)
        self.player = torch.load(player_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.player[idx,:], self.labels[idx]