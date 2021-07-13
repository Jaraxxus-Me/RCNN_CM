"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np
import torchvision
import os
import datetime
import pickle

import transforms
from network_files import Find, AnchorsGenerator
from backbone import MobileNetV2, vgg, resnet101
from finetune_data import MetaDataset, FtDataSet, COCODataSet
from train_utils import get_coco_api_from_dataset, CocoEvaluator
from train_utils.config import cfg
from train_utils import train_eval_utils_meta as utils
from collections import OrderedDict


def create_model(num_classes, phase):
#     # https://download.pytorch.org/models/vgg16-397923af.pth
#     # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
#     # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
#     # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
#     # backbone.out_channels = 512
#     # MobileNet backbone
    backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    backbone.name = "mob"
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = Find(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, phase=phase)
#     # ResNet backbone
    # backbone = resnet101()
    # backbone.name = "res"
    # print("Loading pretrained weights from %s" % ("./backbone/resnet101.pth"))
    # state_dict = torch.load("./backbone/resnet101.pth")
    # backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})
    # backbone.out_channels = 2048

    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))

    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
    #                                                 output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
    #                                                 sampling_ratio=2)  # 采样率

    # model = Find(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler, phase=phase)

    return model

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device testing.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    #dataset config
    coco_root = args.data_path  # VOCdevkit
    if args.meta_type == 1:  #  use the first sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_FIRST
    if args.meta_type == 2:  #  use the second sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_SECOND
    if args.meta_type == 3:  #  use the third sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_THIRD
    if args.meta_type == 4:  #  use the first sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_FORTH
    if args.meta_type == 5:  #  use the second sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_FIFTH
    if args.meta_type == 6:  #  use the third sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_SIXTH
    # check voc root
    # load validation data set the same, 2012+2007 val.txt
    batch_size = args.bs
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    val_data_set = COCODataSet(coco_root, metaclass, data_transform["val"], "val.txt")
    category_index=val_data_set.category_index
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_data_set.collate_fn)

    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    model = create_model(len(metaclass)+1, 2)

    # 载入你自己训练好的模型权重
    weights_path = os.path.join(parser_data.resume_dir,"mob-find-{}-type{}-{}shots.pth".format(parser_data.epoch, parser_data.meta_type, parser_data.shots))
    # weights_path = os.path.join(parser_data.resume_dir,"mobile-find-20.pth")
    print("Loading trained model from {}".format(weights_path))
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model'])
    # load class prototype
    pkl_save_path = os.path.join(parser_data.resume_dir, 'meta_type_{}'.format(parser_data.meta_type))
    pkl_file = open(os.path.join(pkl_save_path,
                                    str(parser_data.epoch) + '_shots_' + str(parser_data.shots) + '_mean_class_attentions.pkl'), 'rb')

    class_proto = pickle.load(pkl_file)
    # print(model)
    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_data_set)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_data_set_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image, class_prototype = class_proto)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)
    model_name="mob-model-{}-type{}-{}shots.pth".format(parser_data.epoch, parser_data.meta_type, parser_data.shots)[:-4]
    # model_name="mobile-find-20.pth"
    # 将验证结果保存至txt文件中
    with open("{}/record_mAP_{}.txt".format(parser_data.output_dir,model_name), "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='/data/', help='dataset root')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume_dir', default='./fine_find_weight/', type=str, help='resume from checkpoint')
    # split (1/2/3)
    parser.add_argument('--meta_type', default=1, type=int,
                        help='which split of VOC to implement, 1, 2, or 3')
    # batch size
    parser.add_argument('--bs', default=8, type=int, metavar='N',
                        help='batch size when validation.')
    # batch size
    parser.add_argument('--output_dir', default="./find_r", metavar='N',
                        help='where to save result')
    parser.add_argument('--epoch', default=29, metavar='N',
                        help='epoch of model to load')
    parser.add_argument('--shots', default=2, metavar='N',
                        help='shots of model to load')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
