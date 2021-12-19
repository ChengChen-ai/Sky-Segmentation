import torch.utils.data as data
from sklearn.utils import shuffle
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
    image_paths_B = []
    path_all = []


    for root, _, fnames in sorted(os.walk(dir)):
        fnames = shuffle(fnames,n_samples=len(fnames))
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths_A.append(path)
                image_paths_B.append(path.replace("images", "labels").replace("jpg", "png"))

    return image_paths_A, image_paths_B


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=True,
                 loader=default_loader):
        img_paths_A, img_paths_B = make_dataset(root)
        if len(img_paths_A) == 0 or len(img_paths_B) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.img_paths_A = img_paths_A
        self.img_paths_B = img_paths_B
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_A = self.img_paths_A[index]
        path_B = self.img_paths_B[index]
        img_A = self.loader(path_A)
        img_B = self.loader(path_B)
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        if self.return_paths:
            return img_A, img_B, path_A, path_B
        else:
            return img_A, img_B, '', ''

    def __len__(self):
        return len(self.img_paths_A)
