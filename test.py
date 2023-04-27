import torch
from PIL import Image
from dataset import MyDataset
from unet.unet_model import UNet
from predict import predict_img

img_dir = "./data/val_imgs/ISIC_0015190.jpg"
full_img = Image.open(img_dir)
scale = 1
img = torch.from_numpy(MyDataset.preprocess(full_img, scale, is_mask=False))
img.unsqueeze_(0)

device = torch.device("cuda")


load_model = UNet()
load_model.to(device=device)
load_model.load_state_dict(torch.load("./checkpoint/checkpoint_epoch10.pth", map_location=device))


x = predict_img(load_model, full_img,device)
print(x)