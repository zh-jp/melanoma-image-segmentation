import logging
import wandb
import time
import torch
from dataset import MyDataset
from unet.unet_model import UNet
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

train_img_dir = "./data/train/imgs/"
train_mask_dir = "./data/train/masks/"

test_img_dir = "./data/test/imgs/"
test_mask_dir = "./data/test/masks/"


@torch.inference_mode()
def test(model,
         device,
         img_dir,
         mask_dir,
         batch_size: int = 10,
         img_scale: float = 1.,
         out_threshold: float = 0.5,
         amp: bool = True):
    model.to(device=device)
    model.eval()
    dataset = MyDataset(img_dir, mask_dir, img_scale)
    loader_args = dict(batch_size=batch_size, num_workers=5,
                       pin_memory=True, drop_last=True)
    dataloader = DataLoader(dataset, shuffle=True, **loader_args)
    num_total = len(dataloader)
    total = 0

    experiment = wandb.init(project="melanoma-image-segmentation", resume="allow", anonymous="must")
    experiment.config.update(
        dict(batch_size=batch_size, img_dir=img_dir, img_scale=img_scale, amp=amp)
    )
    logging.info(f'''Start test
        Batch size:         {batch_size}
        Device:             {device.type}
        Image scale:        {img_scale}
        Image directory:    {img_dir}
        Mixed precision:    {amp}
    ''')
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
        for batch in tqdm(dataloader, total=num_total):
            imgs, masks = batch['img'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.int8)
            outputs = model(imgs)
            outputs = outputs.squeeze(1)
            pred_masks = (torch.sigmoid(outputs) > out_threshold).type_as(masks)
            mean_iou = get_IoU(masks, pred_masks, True)
            total += imgs.shape[0]
            experiment.log({
                "total": total,
                "mean IoU": mean_iou
            })


def get_IoU(tensor1: Tensor, tensor2: Tensor, is_batch: bool = False):
    assert tensor1.shape == tensor2.shape, \
        f"Both should be same size, but tensor1 size is {tensor1.shape} and tensor2 size is {tensor2.shape}"

    def IoU(t1: Tensor, t2: Tensor):
        t1 = t1.view(-1)
        t2 = t2.view(-1)
        # 计算交集和并集
        intersection = (t1 & t2).sum()
        union = (t1 | t2).sum()
        iou = intersection / (union + 1e-6)
        return iou

    if is_batch:
        batch_size = tensor1.shape[0]
        result = 0.
        for i in range(batch_size):
            res = IoU(tensor1[i], tensor2[i])
            result += res
        mean_iou = result / batch_size
        return mean_iou
    return IoU(tensor1, tensor2)


if __name__ == "__main__":
    net = UNet()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load("./checkpoint/2023-05-11-epoch20.pth", map_location=device))
    test(net, device, test_img_dir, test_mask_dir)
