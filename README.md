# Melanoma-image-segmentation
> 黑色素瘤影像分割系统
## 主目录/文件
- front: 基于Flask搭建前端
- part_class: 分类部分数据集、模型搭建与训练
- part_seg: 分割部分数据集、模型搭建与训练 
### `part_class`与`part_seg`均包含以下文件 or 目录：

- dataset.py: 自定义数据集
- evaluate.py: 使用验证集进行模型评估
- predict.py: 调用模型输入单张图片
- test.py: 使用测试集进行测试
- train.py: 使用训练集进行训练
- /checkpoint: 存放模型训练时的checkpoints（被包含在.gitignore）
- /data: 存放原始数据集（被包含在.gitignore）
- /log: 存放日志（被包含在.gitignore）

### 其他文件 or 目录：

- resnet.py: ResNet网络模型文件
- vis_cnn.py: 使用Grad-CAM提取类激活图
- /unet: U-Net网络模型文件
- dice.py: 计算Dice损失函数
- mask_in_img.py: 将掩膜叠加到原始影像

## 相关技术
`PyTorch` `Flask` `ResNet` `U-Net`
## 实现功能
本设计构建了一种基于深度学习的黑色素瘤影像分割系统，并对其分类、分割性能进行评估。
用户仅需要上传皮肤影像，系统将根据相关模型计算结果，将结果反馈至前端。

## 演示
- 界面
![system.png](front%2Fstatic%2Fimages%2Fsystem.png)
- 黑色素瘤
![is_melanoma.png](front%2Fstatic%2Fimages%2Fis_melanoma.png)
- 非黑色素瘤
![not_melanoma.png](front%2Fstatic%2Fimages%2Fnot_melanoma.png)
## 数据集

分类数据：[Kaggle melanoma](https://www.kaggle.com/datasets/drscarlat/melanoma)

分割数据：[2017 ISIC Challenge Datasets](https://challenge.isic-archive.com/data/#2017)

## 参考文献

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
3. [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
4. https://zhuanlan.zhihu.com/p/474790387
5. https://www.jianshu.com/p/9c39e714babb
6. https://www.jianshu.com/p/7f54d8cadeca