import logging

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from os import listdir
from glob import glob
from PIL import Image


def mask2np(idx, mask_dir, mask_suffix):
    """
    掩码转 numpy 数组，0代表黑色， 255代表白色
    :param idx: 图片（指掩膜）文件编号
    :param mask_dir: 图片路径
    :param mask_suffix: 图片后缀名
    :return: np数组
    """
    mask_file = glob(mask_dir + idx + mask_suffix + ".*")[0]
    mask = np.asarray(Image.open(mask_file))
    if mask.ndim == 2:
        return mask
    else:
        raise ValueError(f"Loader masks' dimensions should be 2, found {mask.ndim}.")


class MyDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, scale: float = 1., mask_suffix: str = "_segmentation"):
        """
        构造函数
        :param img_dir: 数据集图片所在目录
        :param mask_dir: 数据集图片掩膜所在目录
        :param scale:  图片缩放比例
        :param mask_suffix: 掩膜后缀名
        """
        assert 0. < scale <= 1., "scale should be 0< scale <= 1"
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [name[:name.rfind('.')] for name in listdir(self.img_dir)]  # 去除文件扩展名
        if not self.ids:
            raise FileNotFoundError(f"There are no images in the {img_dir} directory.")
        logging.info(f"Created dataset with {len(self.ids)} examples.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        # 确定文件名
        name = self.ids[item]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.img_dir.glob(name + ".*"))
        assert len(mask_file) == 1, f'Either no mask found for the id: {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image found for the id: {name}: {img_file}'

        # 加载图片
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        assert mask.size == img.size, f"Image and mask {name}'s size should be same, " \
                                      f"but mask size is {mask.size} and image size is {img.size}."
        return mask, img
