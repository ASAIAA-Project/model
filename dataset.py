import torch
import pickle

import torchvision.transforms as transforms

from torch.utils import data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_NET_MEAN = [0.485, 456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class AVADataset(data.Dataset):
    """AVA dataset
    Args:
        pickle_file: a 4-column pickle_file
            column 0 contains the id of the image
            column 1 contains the name of the image
            column 2 contains the mean of scores of the image
            column 3 contains the std of scores of the image
        root_dir: directory to the images (Path object)
        transform: preprocessing and augmentation of the training images
    """
    def __init__(self, pickle_file, root_dir, wrap_size=224):
        self.root_dir = root_dir
        with open(root_dir / pickle_file, 'rb') as handle:
            self.annotations = pickle.load(handle)

        normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        self.transform = transforms.Compose([
            transforms.Resize((wrap_size, wrap_size)),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = str(self.root_dir / 'images' /
                       f'{int(self.annotations[idx][1])}.jpg')
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        ground_truth = torch.tensor(self.annotations[idx][2:])

        return img_tensor, ground_truth
