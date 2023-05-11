import logging
from resnet import ResNet, BasicBlock, Bottleneck
from densenet import DenseNet
from tqdm import tqdm
from dataset import MyDataset
from torch.utils.data import DataLoader

import torch

train_dir = "./data/train/"
test_dir = "./data/test/"
img_scale = 1.


@torch.inference_mode()
def test(model, dataloader, device, amp: bool = True):
    model.to(device=device)
    model.eval()
    num_total = len(dataloader)
    total = 0
    correct = 0
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for imgs, labels in tqdm(dataloader, total=num_total):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
        print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    # model = ResNet(Bottleneck, [3, 4, 6, 3])
    model = DenseNet(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load("./checkpoint/2023-05-11-epoch20.pth", map_location=device))
    dataset = MyDataset(test_dir, img_scale)

    loader_args = dict(batch_size=5, num_workers=5,
                       pin_memory=True, drop_last=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)

    test(model, train_loader, device)
