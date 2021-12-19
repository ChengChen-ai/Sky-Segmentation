import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    image_paths_A = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths_A.append(path)
    # image_paths = sorted(image_paths_A)
    return image_paths_A


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        img_paths_A = make_dataset(root)
        if len(img_paths_A) == 0 :
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.img_paths_A = img_paths_A
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_A = self.img_paths_A[index]
        img_A = self.loader(path_A)
        if self.transform is not None:
            img_A = self.transform(img_A)
        if self.return_paths:
            return img_A, path_A
        else:
            return img_A, ''

    def __len__(self):
        return len(self.img_paths_A)