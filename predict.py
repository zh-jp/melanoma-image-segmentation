import torch
import torch.nn.functional as F
from dataset import MyDataset
import numpy as np
from PIL import Image


def predict_img(net, full_img, device, scale=1., out_threshold=0.5):
    """

    :param net: 模型
    :param full_img: 待预测图片（Image类型
    :param device: cuda 或 cpu
    :param scale: 图片缩放比例
    :param out_threshold:  sigmoid阈值
    :return:    由 0 1 构成的 numpy 矩阵，大小为uniform（默认512x512）
    """
    net.eval()
    img = torch.from_numpy(MyDataset.preprocess(full_img, scale, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode="bilinear")
        mask = torch.sigmoid(output) > out_threshold
    return mask[0].long().squeeze().numpy()


def mask2img(mask: np.ndarray, outsize: tuple = (512, 512)):  # 将numpy数组转为二值图，255代替1表示黑色
    """

    :param mask: 掩膜的numpy数组
    :param outsize: 输出图像的大小
    :return: Image类型图像
    """
    mask[mask > 0] = 255
    x_img = Image.fromarray(mask.astype('int8'))
    return x_img.resize(outsize)
