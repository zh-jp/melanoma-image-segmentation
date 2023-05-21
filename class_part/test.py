import logging
import wandb
import time
from resnet import ResNet, BasicBlock, Bottleneck
from tqdm import tqdm
from dataset import MyDataset
from torch.utils.data import DataLoader

import torch

train_dir = "./data/train/"
test_dir = "./data/test/"
checkpoint_dir = "./checkpoint/"
log_dir = "./log/"
img_scale = 1.


@torch.inference_mode()
def test(model,
         device,
         img_dir: str = "./data/train/",
         batch_size: int = 10,
         img_scale: float = 1.,
         amp: bool = True):
    model.to(device=device)
    model.eval()

    dataset = MyDataset(img_dir, img_scale)
    loader_args = dict(batch_size=batch_size, num_workers=5,
                       pin_memory=True, drop_last=True)
    dataloader = DataLoader(dataset, shuffle=True, **loader_args)

    num_total = len(dataloader)
    total = 0

    TP, TN, FN, FP = 0, 0, 0, 0
    epsilon = 1e-6

    experiment = wandb.init(project='melanoma-image-segmentation', resume='allow', anonymous='must')
    experiment.config.update(
        dict(batch_size=batch_size, img_dir=img_dir, img_scale=img_scale, amp=amp)
    )
    logging.info(f'''Start test
        Batch size:     {batch_size}
        Device:         {device.type}
        Image scale:    {img_scale}
        Image directory:{img_dir}
        Mixed precision:{amp}
    ''')
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for imgs, labels in tqdm(dataloader, total=num_total):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            for pred, lab in zip(predicted, labels):
                if pred == 0 and lab == 0:
                    TP += 1
                if pred == 1 and lab == 1:
                    TN += 1
                if pred == 0 and lab == 1:
                    FP += 1
                if pred == 1 and lab == 0:
                    FN += 1
            accuracy = (TP + TN) * 1. / (TP + TN + FP + FN + epsilon)
            precision = TP * 1. / (TP + FP + epsilon)
            recall = TP * 1. / (TP + FN + epsilon)

            experiment.log({
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "total": total,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            })


if __name__ == "__main__":
    log_time = time.strftime("%Y-%m-%d", time.localtime())
    log_filename = log_dir + log_time + '-train.log'
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename=log_filename,
                        filemode='a')

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_name = checkpoint_dir + "2023-05-12-epoch20.pth"
    model.load_state_dict(torch.load(load_name, map_location=device))
    test(model, device, img_dir=train_dir)
