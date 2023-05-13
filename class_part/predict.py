import torch
from .dataset import MyDataset


def predict(net, img, device, scale: float = 1.0):
    net.to(device=device)
    net.eval()
    img = torch.from_numpy(MyDataset.preprocess(img, scale))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
    _, index = torch.max(output)
    return index
