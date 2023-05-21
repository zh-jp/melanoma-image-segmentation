from tqdm import tqdm
import torch


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion):
    net.eval()
    total_num = len(dataloader)
    total = 0
    total_loss = 0.
    with torch.autocast(device.type if device.type != "mps" else 'cpu', enabled=amp):
        for imgs, labels in tqdm(dataloader, total=total_num, desc="Validation round", unit="batch", leave=False):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = net(imgs)
            loss = criterion(outputs, labels)
            total += imgs.size(0)
            total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / total
    net.train()
    return avg_loss
