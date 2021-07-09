# FIND_COS_LOSS

## 该项目主要是来自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.6(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见```requirements.txt```

## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是```fasterrcnn_resnet50_fpn_coco.pth```文件，
  不是```fasterrcnn_resnet50_fpn_coco-258fb6c6.pth```
 
 
## 数据集，本例程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 如果不了解数据集或者想使用自己的数据集进行训练，请参考我的bilibili：https://b23.tv/F1kSCK
* 使用ResNet50+FPN以及迁移学习在VOC2012数据集上得到的权重: 链接:https://pan.baidu.com/s/1ifilndFRtAV5RDZINSHj5w 提取码:dsz8

## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
* 若要使用多GPU训练，使用```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py```指令,```nproc_per_node```参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上```CUDA_VISIBLE_DEVICES=0,3```(例如我只要使用设备中的第1块和第4块GPU设备)
* ```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py```

## Instruction
This branch is used to validate and adjust the cos_similarity loss function. Regression branch remains the same as baseline.