import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.image_folder_test import ImageFolder
import argparse
from collections.abc import *
import torchvision
import matplotlib.pyplot as plt
import numpy as np



class PairedData(object):
    def __init__(self, data_loader_A):
        self.data_loader_A = data_loader_A

    def __iter__(self):
        self.data_loader_A_iter = iter(self.data_loader_A)
        return self

    def __next__(self):
        A, A_paths = next(self.data_loader_A_iter)
        return {'A': A, 'A_paths': A_paths}


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
            shuffle=params.shuffle)

        self.dataset_A = dataset_A
        self.paired_data = PairedData(self.dataset_A)
        print(isinstance(self.paired_data, Iterator))

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_A)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data', type=str)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--load_size', default=142, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--phase', default='train', type=str)
    params = parser.parse_args()

    unalignedDataLoader = UnalignedDataLoader(params)
    dataset = unalignedDataLoader.load_data()
    for _, u in enumerate(dataset):
        img_A = torchvision.utils.make_grid(u['B']).numpy()
        fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(13,7))
        axes[0].imshow(np.transpose(img_A, (1, 2, 0)))
        plt.show()


