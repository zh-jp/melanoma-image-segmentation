import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, is_a_batch: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not is_a_batch  # 要么维度是3（代表是个batch），要么是单个图片

    # 确认计算维度，如果是单个图片（dim=2， 不是一个batch）采用前者，反之采用后者
    sum_dim = (-1, -2) if input.dim() == 2 or not is_a_batch else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)  # 求交集
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)  # 求并集
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)  # 并集为0时，将交集赋值给并集
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor):
    fn = dice_coeff
    return 1 - fn(input, target, is_a_batch=True)
