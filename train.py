import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dice import dice_loss
from dataset import MyDataset
import torch.nn.functional as F
from unet.unet_model import UNet
to_tensor = transforms.ToTensor()

img_dir = './data/imgs/'
mask_dir = './data/masks/'
checkpoint_dir = './checkpoint/'
mask_suffix = '_segmentation'


def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 10,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_check_point: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0
):
    # 1. 创建数据集
    dataset = MyDataset(img_dir, mask_dir, img_scale, mask_suffix)

    # 2. 划分训练集、验证集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.manual_seed(0))

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count() - 3,
                       pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # 4. 设置优化器、损失、学习率调整策略
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['img'], batch['mask']
                assert imgs.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels.'

                imgs = imgs.to(device=device, dtype=torch.float32, memory_foramt=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.int8)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(imgs)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float)
                    else:
                        # 目前只作为影响分割，不考虑多分类
                        raise ValueError("The classes is too many")
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                global_step += 1
                epoch_loss += loss.item()

                pbar.update(imgs.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                divison_step = (n_train // (5 * batch_size))

        if save_check_point and (epoch % 10 == 0 or epoch == epochs):
            static_dict = model.state_dict()
            name = checkpoint_dir + 'checkpoint_epoch{}.pth'.format(epoch)
            torch.save(static_dict, name)
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == "__main__":
    model = UNet(n_channels=3, n_classses=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    train_model(model, device)
