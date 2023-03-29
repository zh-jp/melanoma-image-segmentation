# melanoma-image-segmentation
基于黑色素瘤图像，建立黑色素瘤图像分割模型

[数据集](https://www.kaggle.com/datasets/drscarlat/melanoma)</br>
[数据集](https://challenge.isic-archive.com/data/#2017)
## 已读文献
>[1]王嘉仪,陈晓晴,吴传旭等.基于深度学习的“智能医学影像分割系统”的开发与实现[J].电子技术与软件工程,2022,No.222(04):177-180. </br>
[2]龙雪,李政林,王智文,呼和乌拉.基于改进U-Net网络的肺部CT图像结节分割方法[J].广西科技大学学报,2022,33(01):63-70+77.DOI:10.16375/j.cnki.cn45-1395/t.2022.01.010.
## Suggestion from ChatGPT
恭喜你开始了你的毕业设计！下面是建议你开始的步骤：

0. 阅读相关文献和研究资料：在开始你的设计之前，了解黑色素瘤图像分割的研究现状是非常重要的。阅读相关的论文、书籍、技术报告、专利，以及网上的博客、文章，对你的研究非常有帮助。

1. 收集数据集：为了建立黑色素瘤图像分割模型，你需要一个足够的数据集。你可以使用公共的数据集，例如 ISIC（International Skin Imaging Collaboration）数据集。如果这些数据集不符合你的需求，你需要收集和标记自己的数据集。这需要相当的时间和精力，因此要提前计划好。

2. 研究和选择合适的图像分割模型：有许多图像分割模型可以选择，例如传统的阈值分割、边缘检测、区域生长、基于能量函数的分割、基于机器学习的分割等等。针对黑色素瘤图像分割问题，建议选择一些基于深度学习的图像分割模型，例如 U-Net、Mask R-CNN、DeepLab 等。

3. 实现图像分割模型：使用 Python 和深度学习框架，例如 TensorFlow 或 PyTorch，实现你选择的图像分割模型。在训练模型之前，你需要对数据集进行预处理、数据增强等操作。

4. 验证和评估模型：训练模型后，你需要对模型进行验证和评估。使用测试数据集对模型进行测试，评估模型的性能和准确性。

5. 开发黑色素瘤影像分割系统：在模型验证和评估之后，你需要开发黑色素瘤影像分割系统。使用 Python 和相关的库和框架，例如 Flask、Django、FastAPI、Streamlit 等，实现你的系统。你需要设计和实现系统的前端界面、后端服务、数据库等功能，以及集成图像分割模型。

6. 测试和部署系统：在开发完系统之后，进行测试，确保系统能够正确地接收黑色素瘤图像，并输出黑色素瘤图像分割结果。在完成测试后，你需要将系统部署到服务器上，使其能够在真实环境中运行。

这是一个大致的指南，但是具体的步骤可能因为不同的情况有所不同。希望这些步骤能够对你的毕业设计提供帮助！

</br>

如果您想要在pytorch中使用Mask R-CNN模型，您可以参考以下几个资源：

Torchvision提供了一个Mask R-CNN模型的实现，您可以使用它来创建、训练或加载预训练的模型1。
https://pytorch.org/vision/master/models/mask_rcnn.html </br>
LearnOpenCV网站有一个教程，介绍了如何使用pytorch和torchvision来进行实例分割2。
https://learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/ </br>
Analytics Vidhya网站也有一个教程，介绍了如何使用pytorch和torchvision来构建、训练和测试Mask R-CNN模型3。</br>
https://www.analyticsvidhya.com/blog/2023/02/mask-r-cnn-for-instance-segmentation-using-pytorch/