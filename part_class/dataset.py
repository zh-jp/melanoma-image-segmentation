import logging

from PIL import (Image, ImageFile)
from pathlib import Path
from torch.utils.data import Dataset
from os import listdir
import numpy as np
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
uniform = (224, 224)


class MyDataset(Dataset):
    def __init__(self, img_dir: str, scale: float = 1.):
        assert 0. < scale <= 1., "scale should be 0< scale <=1"
        self.img_dir = Path(img_dir)
        self.scale = scale
        names = listdir(self.img_dir)
        self.classes = {}
        for index, name in enumerate(names):
            self.classes[name] = index

        if not self.classes or len(self.classes) == 0:
            raise FileNotFoundError(f"There are no images in {self.img_dir} directory.")

        self.ids = []
        for name in names:
            filepath = Path(self.img_dir, name)
            for filename in listdir(filepath):
                self.ids.append((Path(name, filename), self.classes[name]), )
        logging.info(f"Create dataset with {len(self.ids)} examples.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        name, label = self.ids[item]
        img = Image.open(Path(self.img_dir, name))
        img = self.preprocess(img, self.scale)
        img = torch.as_tensor(img.copy(), dtype=torch.float32).contiguous()
        return img, label

    @staticmethod
    def preprocess(img, scale):
        img = img.resize(uniform)
        w, h = img.size
        w_, h_ = int(w * scale), int(h * scale)
        assert w_ > 0 and h_ > 0, f"Scale {scale} is too small, images resized would have no pixel."
        img = img.resize((w_, h_), resample=Image.NEAREST)
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return img
