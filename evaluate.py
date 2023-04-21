import torch
from tqdm import tqdm
import torch.nn.functional as F
from dice import dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batchs = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batchs, desc='Validation round', unit='batch', leave=False):
            img, mask_true = batch['img'], batch['mask']
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.int8)

            mask_pred = net(img)
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0,1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, is_a_batch=True)

    net.train()
    return dice_score / max(num_val_batchs, 1)
