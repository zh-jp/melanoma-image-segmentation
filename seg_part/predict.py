import torch
import torch.nn.functional as F
from .dataset import MyDataset
import numpy as np
from PIL import Image


def predict(net, full_img: Image, device, scale=1., out_threshold=0.5):
    """

    :param net: 模型
    :param full_img: 待预测图片（Image类型
    :param device: cuda 或 cpu
    :param scale: 图片缩放比例
    :param out_threshold:  sigmoid阈值
    :return:    由 0 1 构成的 numpy 矩阵，大小为 full_img 的尺寸
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


def mask2img(mask: np.ndarray):  # 将numpy数组转为二值图，255代替1表示黑色
    """

    :param mask: 掩膜的numpy数组
    :return: Image类型图像
    """
    mask[mask > 0] = 255
    x_img = Image.fromarray(mask.astype('int8'))
    return x_img


def get_IoU(img1: Image, img2: Image):
    assert img1.size == img2.size, \
        f"Both should be same size, but image1 size is {img1.size} and image2 size is {img2.size}"

    threshold = 127
    img1_n = np.array(img1)
    img2_n = np.array(img2)
    binary_img1 = np.where(img1_n > threshold, 1, 0)
    binary_img2 = np.where(img2_n > threshold, 1, 0)

    # 计算交集和并集
    intersection = np.logical_and(binary_img1, binary_img2)
    union = np.logical_or(binary_img1, binary_img2)

    # 计算交集和并集的面积
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    # 计算交并比
    iou = intersection_area / union_area

    return iou

