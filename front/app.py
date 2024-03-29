from part_cls.resnet import ResNet, Bottleneck
import part_cls.predict as resnet_predict
from part_seg.unet.unet_model import UNet
import part_seg.predict as unet_predict
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from part_seg.mask_in_img import mask_in_img
from PIL import Image
import uuid

# 上传文件到服务器指定到文件夹中
# Flask 默认静态文件存放在 static 文件夹下，其他文件夹下的文件会导致无法访问
UPLOAD_FOLDER = 'static/tmp/'
ALLOW_FILE = {'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_FOLDER

resnet_state_dict_name = "../part_cls/checkpoint/2023-05-12-epoch20.pth"
unet_state_dict_name = "../part_seg/checkpoint/2023-05-11-epoch20.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2).to(device=device)
resnet.load_state_dict(torch.load(resnet_state_dict_name, map_location=device))
unet = UNet().to(device=device)
unet.load_state_dict(torch.load(unet_state_dict_name, map_location=device))

target = 0  # 代表黑色素瘤的可能


def allow_file(filename: str):
    return '.' in filename and filename.split(".")[-1] in ALLOW_FILE


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_images():
    if os.path.exists(UPLOAD_FOLDER) is False:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if request.method == 'POST':
        file = request.files['photo']
        if file and allow_file(file.filename):
            filename = secure_filename(file.filename)
            url1 = os.path.join(UPLOAD_FOLDER, filename)
            url1 = os.path.normpath(url1)
            file.save(url1)
            res = analysis(url1)
            if res is None:
                return render_template('index.html', img_error=True)
            if isinstance(res, str):
                return render_template('index.html', notmelanoma=True)
            else:
                filename = str(uuid.uuid1()) + '.png'
                url2 = os.path.join(UPLOAD_FOLDER, filename)
                url2 = os.path.normpath(url2)
                res.save(url2)
                url3 = os.path.join(UPLOAD_FOLDER, str(uuid.uuid1()) + '.png')
                url3 = os.path.normpath(url3)
                mask_in_img(url1, url2).save(url3)
                return render_template('index.html', url1=url1, url2=url2, url3=url3)
        return render_template('index.html', img_error=True)


def analysis(img_dir: str):
    try:
        img = Image.open(img_dir)
    except FileNotFoundError:
        return None
    index = resnet_predict.predict(net=resnet, img=img, device=device)
    if index == target:
        pred_mask = unet_predict.mask2img(unet_predict.predict(net=unet, full_img=img, device=device))
        # 不加这一段保存的图片是全黑的
        pred_mask = pred_mask.convert("L")
        pred_mask = pred_mask.point(lambda x: 255 if x > 128 else 0)
        return pred_mask
    return "This picture may not be melanoma."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
