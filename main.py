import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
from mingpt.model import GPT
from mingpt.trainer import Trainer
import pickle

set_seed(8736)

class SortDataset(Dataset):
    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = 6
        self.num_digits = num_digits

    def __len__(self):
        return 5 # default 10000
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        return self.length * 2 -1

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    continue
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train'
            if inp_split == self.split:
                break
        sol = torch.sort(inp)[0]

        cat = torch.cat((inp, sol), dim=0)

        x, y = cat[:-1].clone(), cat[1:].clone()
        y[:self.length-1] = -1
        return x, y


train_dataset = SortDataset('train')
test_dataset = SortDataset('test')

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4
train_config.max_iters = 2000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter time {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()