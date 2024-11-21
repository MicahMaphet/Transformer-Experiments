import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed

set_seed(8736)

class SortDataset(Dataset):
    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = 6
        self.num_digits = num_digits

    def __len__(self):
        return 5 # default 10000

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            return inp


train_dataset = SortDataset('train')
test_dataset = SortDataset('test')
print(train_dataset[10001])