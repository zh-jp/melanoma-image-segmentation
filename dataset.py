import logging
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from os import listdir
import torch
from PIL import Image


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
        # 如果未找到文件则报错
        assert len(mask_file) == 1, f'Either no mask found for the id: {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image found for the id: {name}: {img_file}'

        # 加载图片
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        # 如果图片和掩膜的大小不一致则报错
        assert mask.size == img.size, f"Image and mask {name}'s size should be same, " \
                                      f"but mask size is {mask.size} and image size is {img.size}."

        # 将图片转为numpy数组
        mask = self.preprocess(mask, self.scale, is_mask=True)
        img = self.preprocess(img, self.scale, is_mask=False)

        # 将numpy数组转为张量，并确定类型
        mask = torch.as_tensor(mask.copy(), dtype=torch.int8).contiguous()
        img = torch.as_tensor(img.copy(), dtype=torch.float32).contiguous()
        return {
            "img": img,
            "mask": mask
        }

    @staticmethod
    def preprocess(img, scale, is_mask):
        w, h = img.size
        w_, h_ = int(w * scale), int(h * scale)
        assert w_ > 0 and h_ > 0, f"Scale {scale} is too small, images resized would have no pixel."
        """
        NEAREST插值法是一种图像插值方法，它是一种最简单的插值方法，它将距离目标像素最近的已知点的像素值直接赋给目标像素。
        这种方法的优点是速度快，缺点是图像质量较差。
        
        BICUBIC插值法是一种图像插值方法，它是一种比较常用的插值方法，它利用待求像素坐标反变换后得到的浮点坐标周围的16个邻近像素，
        通过三次多项式S(x)来拟合数据，从而得到插值结果。
        """
        img = img.resize((w_, h_), resample=Image.NEAREST if is_mask is True else Image.BICUBIC)
        img = np.array(img)
        if is_mask:
            img[img == 255] = 1
        else:
            img = img.transpose((2, 0, 1))  # 将通道数移动到第一个下标
            if (img > 1).any():
                img = img / 255.0
        return img
