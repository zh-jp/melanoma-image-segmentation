import cv2
import torch
import uuid
from PIL import Image
from part_class.dataset import MyDataset
from part_class.resnet import ResNet, Bottleneck
import numpy as np
import matplotlib as mpl

checkpoint_dir = "./checkpoint/"


class ShowGradCam:
    def __init__(self, conv_layer):
        assert isinstance(conv_layer, torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def show_on_img(self, input_img):
        '''
        write heatmap on target img
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        if isinstance(input_img, str):
            input_img = cv2.imread(input_img)
        img_size = (input_img.shape[1], input_img.shape[0])
        fmap = self.feature_res[0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        cam = self.gen_cam(fmap, grads_val)
        cam = cv2.resize(cam, img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) / 255.
        cam = heatmap + np.float32(input_img / 255.)
        cam = cam / np.max(cam) * 255
        filename = str(uuid.uuid1()) + ".jpg"
        cv2.imwrite(filename, cam)
        return filename


def comp_class_vec(output_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(output_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output_vec.to("cpu"))  # one_hot = 11.8605

    return class_vec


def vis_cnn(net, device, img_dir: str, layer):
    img = MyDataset.preprocess(Image.open(img_dir), 1.)
    input_tensor = torch.from_numpy(img)
    input_tensor = input_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    net.to(device=device)
    gradCam = ShowGradCam(layer)

    # forward
    output = net(input_tensor)

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()
    filename = gradCam.show_on_img(img_dir)
    return filename


if __name__ == "__main__":
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_name = checkpoint_dir + "2023-05-12-epoch20.pth"
    model.load_state_dict(torch.load(load_name))
    img_path = "./data/test/Melanoma/AUG_0_71.jpeg"
    filename = vis_cnn(model, device, img_path, model.layer4)
    print(filename)
