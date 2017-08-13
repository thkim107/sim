from __future__ import print_function
import os
from glob import glob
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filters("ignore")

# Chroma dataset
# root_dir can vary according to the dataset we want to use(CENS or naive chroma)
class chromaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise Exception("[!] {} not exists.".format(self.root_dir))

        self.paths = glob(os.path.join(self.root_dir, '*.csv'))
        if len(self.paths) == 0:
            raise Exception("[!] No data is found in {}".format(self.root_dir))

        self.chromas = []
        for path in self.paths:
            chroma = pd.read_csv(path)
            self.chromas.append(chroma)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        chroma = self.chromas[index].as_matrix().astype('float')
        out = self.transform(chroma)
        return out

    def __len__(self):
        return len(self.paths)

# Data loader (will be used in main.py)
def get_loader(root_dir, batch_size, num_workers=1, shuffle=True):
    print("[*] Loading Data from {}...".format(root_dir))
    chroma_data = chromaDataset(root_dir)
    data_loader = DataLoader(dataset=chroma_data, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    return data_loader
