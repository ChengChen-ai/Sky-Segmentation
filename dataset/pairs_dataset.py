import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.image_folder import ImageFolder
from collections.abc import *

class PairedData(object):
    def __init__(self, data_loader_A):
        self.data_loader_A = data_loader_A

    def __iter__(self):
        self.data_loader_A_iter = iter(self.data_loader_A)
        return self

    def __next__(self):
        A, B, A_paths ,B_paths = next(self.data_loader_A_iter)
        return {'A': A, 'A_paths': A_paths,
                'B': B, 'B_paths': B_paths}


class UnalignedDataLoader(object):
    def __init__(self, params):
        transform = transforms.Compose([
            transforms.Resize(size=(params.height, params.width)),
            # transforms.Scale(size=(params.load_size, params.load_size)),
            # transforms.RandomCrop(size=(params.height, params.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        dataset_A = torch.utils.data.DataLoader(
            ImageFolder(root=params.data_root + '/' + 'A', transform=transform),
            num_workers=params.num_workers,
            batch_size=params.batch_size,
            shuffle=params.shuffle)

        self.dataset_A = dataset_A
        self.paired_data = PairedData(self.dataset_A)
        print(isinstance(self.paired_data, Iterator))

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_A)