# Melanoma-image-segmentation
## 摘要
黑色素瘤是临床上最为严重、死亡率最高的皮肤癌。但是黑色素瘤不易诊断且对医护人员要求较高。基于深度学习建立图像分割系统代替手动分割可以降低诊断难度、提高诊断效率。
因此，本设计构建了一种基于深度学习的黑色素瘤影像分割系统，实现黑色素瘤的分类和分割。本设计中，分类数据集为HAM10000医学影像数据集，分割数据集选自2017年ISIC公开挑战赛数据集。分类模型选择使用ResNet50；分割模型选择使用U-Net。系统采用轻量级Python Web开发框架Flask设计前端并对接模型。本设计中，ResNet50模型训练结果在测试集上的准确率、精确率、召回率以及F1分数分别为84.69%、86.12%、83.75%以及84.6%，在训练集上的结果分别为86.6%、86.4%、86.71%以及86.58%；
U-Net模型在测试集和训练集上的平均交并比分别为59.46%和67.89%。通过测试用户界面，能够正常调用模型并获取结果。本设计构建的系统能够实现黑色素瘤分类和分割，辅助医生诊断。
## 实现功能
本设计构建了一种基于深度学习的黑色素瘤影像分割系统，并对其分类、分割性能进行评估。
用户仅需要上传皮肤影像，系统将根据相关模型计算结果，将结果反馈至前端。
## 主目录/文件介绍
- front: 基于Flask搭建前端
- part_cls: 分类部分数据集、模型搭建与训练
- part_seg: 分割部分数据集、模型搭建与训练 

**`part_cls`与`part_seg`均包含以下文件 or 目录：**

- dataset.py: 自定义数据集
- evaluate.py: 使用验证集进行模型评估
- predict.py: 调用模型输入单张图片
- test.py: 使用测试集进行测试
- train.py: 使用训练集进行训练
- /checkpoint: 存放模型训练时的checkpoints（被包含在.gitignore）
- /data: 存放原始数据集（被包含在.gitignore）
- /log: 存放日志（被包含在.gitignore）

**其他文件 or 目录：**
- resnet.py: ResNet网络模型文件
- vis_cnn.py: 使用Grad-CAM提取类激活图
- /unet: U-Net网络模型文件
- dice.py: 计算Dice损失函数
- mask_in_img.py: 将掩膜叠加到原始影像

## 相关技术
- 深度学习框架：PyTorch
- 界面前后端框架：Flask
- 深度学习模型：ResNet50、U-Net
- 程序运行数据记录：wandb




## 数据集
**分类数据：[Kaggle melanoma](https://www.kaggle.com/datasets/drscarlat/melanoma)**

| 下载的数据集的目录 | 本项目存放数据集的目录          |
|-----------|----------------------|
| test      | /part_cls/data/test  |
| train     | /part_cls/data/train |
| valid     | /part_cls/data/train |
>在本项目，将验证集valid 与训练集train合并，再按照8:2划分为训练集和验证集

**分割数据：[2017 ISIC Challenge Datasets](https://challenge.isic-archive.com/data/#2017)**

| 下载项                                                                                                                                                                         | 本项目存放数据集的目录                |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| 《Test Data》：[Download (5.4GB)](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip)                                                                | /part_seg/data/test/imgs   |
| 《Test Ground Truth》：[Download (18MB)](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip)                                            | /part_seg/data/test/masks  |
| 《Training Data》：[Download (5.8GB)2000 lesion images in JPEG ...](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip)                             | /part_seg/data/train/imgs  |
| 《Training Ground Truth》：[Download (9MB) 2000 binary mask images in PNG format.](https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip) | /part_seg/data/train/masks |
>zip 文件下载解压后放入对应的文件夹即可。

## 本项目用到的训练后的checkpoint
下载地址：
```txt
链接：https://pan.baidu.com/s/1NqWjX994ANx3tToGKGAetg?pwd=qjsk 
提取码：qjsk
```
分别存放于 `/part_cls/checkpoint` 与 `/part_seg/checkpoint`


## 界面截图
- 界面
![system.png](front%2Fstatic%2Fimages%2Fsystem.png)
- 黑色素瘤
![is_melanoma.png](front%2Fstatic%2Fimages%2Fis_melanoma.png)
- 非黑色素瘤
![not_melanoma.png](front%2Fstatic%2Fimages%2Fnot_melanoma.png)

## 参考文献
1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
3. [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
4. https://zhuanlan.zhihu.com/p/474790387
5. https://www.jianshu.com/p/9c39e714babb
6. https://www.jianshu.com/p/7f54d8cadeca
