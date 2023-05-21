import torch
from part_class import dataset
from PIL import Image


def predict(net, img: Image, device, scale: float = 1.0) -> int:
    net.to(device=device)
    net.eval()
    img = torch.from_numpy(dataset.MyDataset.preprocess(img, scale))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
    _, index = torch.max(output, dim=1)
    return index.item()


