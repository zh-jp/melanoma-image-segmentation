import logging
import time

import torch
import wandb
from tqdm import tqdm
from torch import optim, nn
from resnet import ResNet, BasicBlock
from dataset import MyDataset
from torch.utils.data import DataLoader, random_split
from evaluate import evaluate

img_dir = './data/train/'
checkpoint_dir = './checkpoint/'


def train_model(
        model,
        device,
        epochs: int = 10,
        batch_size: int = 5,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0
):
    model.to(device=device)
    # 1. 创建数据集
    dataset = MyDataset(img_dir, img_scale)

    # 2. 划分数据集、训练集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.manual_seed(0))

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=5,
                       pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # (使用wandb记录模型的训练过程)
    experiment = wandb.init(project='melanoma-image-segmentation', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )
    logging.info(f'''Start train:
        Epoch:          {epochs}
        Batch size:     {batch_size}
        Learning rate:  {learning_rate}
        Training size:  {n_train}
        Validation size:{n_val}
        Checkpoints:    {save_checkpoint}
        Device:         {device.type}
        Imaging:        {img_scale}
        Mixed precision:{amp}
    ''')

    # 4. 设置优化器、损失、学习率调整策略
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    division_step = (n_train // (5 * batch_size))

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for imgs, labels in train_loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)

                with torch.autocast(device.type if device != 'mps' else 'cpu', enabled=amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                global_step += 1
                epoch_loss += loss.item()

                pbar.update(imgs.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                experiment.log({
                    "train loss": loss.item(),
                    "step": global_step,
                    "epoch": epoch
                })

                if division_step > 0 and global_step % division_step == 0:
                    val_loss = evaluate(model, val_loader, device, amp, criterion)
                    scheduler.step(val_loss)
        if save_checkpoint:
            save_time = time.strftime("%Y-%m-%d", time.localtime())
            static_dict = model.state_dict()
            name = checkpoint_dir + save_time + "-epoch{}.pth".format(epoch)
            torch.save(static_dict, name)
            logging.info(f"Checkpoint {epoch} saved!")


if __name__ == "__main__":
    log_time = time.strftime("%Y-%m-%d", time.localtime())
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename=(log_time + '-train.log'),
                        filemode='a')
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, device)
