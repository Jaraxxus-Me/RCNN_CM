import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
from train_utils.config import cfg
from torchvision import transforms
import transforms as data_t
from finetune_data_coco import MetaDataset, FtDataSet, COCODataSet
from finetune_data_subt import MetaData, FtData, SubTData
from network_files import Find, AnchorsGenerator
from backbone import MobileNetV2, vgg, resnet101
from draw_box_utils import draw_box


def create_model(phase):
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


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device testing.".format(device.type))
    data_transform = {
        "val": data_t.Compose([transforms.ToTensor()])
    }

    #dataset config
    if args.dataset=="coco":
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
        print('loading {:s} dataset for visulazation'.format(args.dataset))
        val_data_set = COCODataSet(coco_root, metaclass, data_transform["val"], "val.txt")
        category_index=val_data_set.category_index
    else:
        subt_root = args.data_path  # VOCdevkit
        json_info = os.path.join(subt_root,"Type_Info","SUBT_type_{:s}.json".format(args.dataset[-1]))
        with open(json_info, "r") as f:
            info_dict = json.load(f)
            metaclass = info_dict["classes"]
        # load validation data set the same, 2012+2007 val.txt
        print('loading {:s} dataset for visulazation'.format(args.dataset)) 
        val_data_set = SubTData(subt_root, metaclass, data_transform["val"], "val_{:s}.txt".format(args.dataset[-1]))
        category_index=val_data_set.category_index
    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    model = create_model(2)

    # 载入你自己训练好的模型权重
    if args.finetuned:
        if parser_data.dataset=="coco":
            weights_path = os.path.join(parser_data.resume_dir,"mob-find-{}-type{}-{}shots.pth".format(parser_data.epoch, parser_data.meta_type, parser_data.shots))
        else:
            weights_path = os.path.join(parser_data.resume_dir,"mob-find-{}-type{}-{}shots.pth".format(parser_data.epoch, parser_data.dataset[-1], parser_data.shots))
    else:
        weights_path = os.path.join("./find_weights","mobile-find-20.pth")
    # weights_path = os.path.join(parser_data.resume_dir,"mobile-find-20.pth")
    print("Loading trained model from {}".format(weights_path))
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model'])
    # load class prototype
    if parser_data.dataset=="coco":
        pkl_save_path = os.path.join(parser_data.resume_dir, 'meta_type_{}'.format(parser_data.meta_type))
    else:
        pkl_save_path = os.path.join(parser_data.resume_dir, 'meta_type_{}'.format(parser_data.dataset[-1]))
    pkl_file = open(os.path.join(pkl_save_path,
                                    str(parser_data.epoch) + '_shots_' + str(parser_data.shots) + '_mean_class_attentions.pkl'), 'rb')

    class_proto = pickle.load(pkl_file)
    # print(model)
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        for index in tqdm(range(len(val_data_set.xml_list)), desc="validation..."):
            # img preparation
            img_path = os.path.join(val_data_set.img_root, val_data_set.xml_list[index].split("/")[-1][:-4]+".jpg")
            original_img = Image.open(img_path)
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # img_height, img_width = img.shape[-2:]
            # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            # model(init_img)

            # t_start = time_synchronized()
            predictions = model(img.to(device), class_prototype=class_proto)[0]
            # t_end = time_synchronized()
            # print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(original_img,
                    predict_boxes,
                    predict_classes,
                    predict_scores,
                    category_index,
                    thresh=args.thresh,
                    line_thickness=3)
            plt.imshow(original_img)
            plt.show()
            # 保存预测的图片结果
            out_vis = os.path.join(args.output_dir, args.dataset+"_"+str(args.meta_type))
            if not os.path.isdir(out_vis):
                os.mkdir(out_vis)
            original_img.save(os.path.join(args.output_dir, args.dataset+"_"+str(args.meta_type), val_data_set.xml_list[index].split("/")[-1][:-4]+".jpg"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--dataset', default='coco', help='dataset:coo or subt')
    parser.add_argument('--data_path', default='/home/user/ws/dataset/coco', help='dataset root')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume_dir', default='./fine_find_weight/', type=str, help='resume from checkpoint')
    parser.add_argument('--finetuned', default=False, help='if finetuned?')
    # split (1/2/3)
    parser.add_argument('--meta_type', default=1, type=int,
                        help='which split of VOC to implement, 1, 2, or 3')
    parser.add_argument('--output_dir', default="./find_r", metavar='N',
                        help='where to save result')
    parser.add_argument('--epoch', default=9, metavar='N',
                        help='epoch of model to load')
    parser.add_argument('--shots', default=1, metavar='N',
                        help='shots of model to load')
    parser.add_argument('--thresh', default=0.5, type=float,
                        help='shots of model to load')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)

