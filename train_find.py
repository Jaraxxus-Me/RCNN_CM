import os
import datetime

import torch
import torchvision
from collections import OrderedDict
import transforms
from network_files import Find, AnchorsGenerator
from backbone import MobileNetV2, vgg, resnet101
from my_dataset import VOCDataSet
from metadata import MetaDataset
from train_utils import train_eval_utils_meta as utils
from train_utils.config import cfg

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


def main(args):
    #config device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    if not os.path.exists("find_r"):
        os.makedirs("find_r")
    results_file = "find_r/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path  # VOCdevkit
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load dataset
    if args.phase == 1:
        # First phase only use the base classes, each class has 200 class data
        shots = 2
        
        if args.meta_type == 1:
            args.train_txt = "voc_2007_train_first_split+voc_2012_train_first_split"
            metaclass = cfg.TRAIN.BASECLASSES_FIRST
            allclass = cfg.TRAIN.ALLCLASSES_FIRST
        elif args.meta_type == 2:
            args.train_txt = "voc_2007_train_second_split+voc_2012_train_second_split"
            metaclass = cfg.TRAIN.BASECLASSES_SECOND
            allclass = cfg.TRAIN.ALLCLASSES_SECOND
        elif args.meta_type == 3:
            args.train_txt = "voc_2007_train_third_split+voc_2012_train_third_split"
            metaclass = cfg.TRAIN.BASECLASSES_THIRD
            allclass = cfg.TRAIN.ALLCLASSES_THIRD
    else:
        # Second phase only use fewshot number of base and novel classes
        shots = args.shots
        if args.meta_type == 1:  #  use the first sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_FIRST
            args.train_txt = "voc_2007_train_first"
        if args.meta_type == 2:  #  use the second sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_SECOND
            args.train_txt = "voc_2007_train_second"
        if args.meta_type == 3:  #  use the third sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_THIRD
            args.train_txt = "voc_2007_train_third"

        # load train data set
    # VOCdevkit -> VOC2012/VOC2007 -> ImageSets -> Main -> train.txt
    # new dataset, combine VOC2012/VOC2017, train.txt, 8218 images
    train_data_set = VOCDataSet(VOC_root, metaclass, data_transform["train"], args.train_txt)
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.bs
    val_size = args.bs_v
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012/2007 -> ImageSets -> Main -> val.txt
    val_data_set = VOCDataSet(VOC_root, metaclass, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=val_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)

    # construct the input dataset of cpro network
    # meta training use data from voc2007+2012
    if args.phase == 1:
        img_set = [('2007', 'trainval'), ('2012', 'trainval')]
    # meta fine-tune use data from voc2007 only
    else:
        img_set = [('2007', 'trainval')]
    metadataset = MetaDataset(VOC_root, img_set, metaclass,
                                    shots=shots, shuffle=True)
    metaloader = torch.utils.data.DataLoader(metadataset, batch_size=1,
                                                shuffle=False, num_workers=0, pin_memory=True)



    # create model num_classes equal background + meta classes
    # model = create_model(len(metaclass)+1, 1)
    model = create_model(1)
    # print(model)

    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []
    # resnet backbone training schedule
    if model.backbone.name=="res":
        lr = cfg.TRAIN.LEARNING_RATE
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        # params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.TRAIN.MOMENTUM)
        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.33)

        for epoch in range(args.start_epoch, args.epochs, 1):
            # train for one epoch, printing every 50 iterations
            mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader, metaloader,
                                                device, epoch, print_freq=50, cls_w=args.cls, metabs=args.metabs)
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)

            # update the learning rate
            lr_scheduler.step()

            # evaluate on the test dataset
            coco_info = utils.evaluate(model, val_data_set_loader, metaloader, 2, device=device)

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # pascal mAP

            # save weights
            # 仅保存最后10个epoch的权重
            if epoch >= 4:
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch}
                torch.save(save_files, os.path.join(args.output_dir,"resnet101-find-{}.pth".format(epoch)))
        # mobile net training schedule
    else:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  first frozen backbone and train 5 epochs                   #
        #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        for param in model.backbone.parameters():
            param.requires_grad = False

        # define optimizer
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'roi_heads.simi_head' in key:
                    params += [{'params': [value], 'lr': 0.01}]
                else:
                    params += [{'params': [value]}]
        # params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.008,
                                    momentum=0.9, weight_decay=0.0005)

        init_epochs = 5
        if args.resume=="":
            for epoch in range(init_epochs):
                # train for one epoch, printing every 10 iterations
                mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader, metaloader,
                                                device, epoch, print_freq=50, cls_w=args.cls, metabs=args.metabs, warmup=True)
                train_loss.append(mean_loss.item())
                learning_rate.append(lr)

            # evaluate on the test dataset
            coco_info, pro = utils.evaluate(model, val_data_set_loader, metaloader, 2, device=device)

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")


            val_map.append(coco_info[1])  # pascal mAP

            torch.save(model.state_dict(), "{}/pretrain.pth".format(args.output_dir))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  second unfrozen backbone and train all network     #
        #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # 冻结backbone部分底层权重
        for name, parameter in model.backbone.named_parameters():
            split_name = name.split(".")[0]
            if split_name in ["0", "1", "2", "3"]:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

        # define optimizer
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'roi_heads.simi_head' in key:
                    params += [{'params': [value], 'lr': 0.01}]
                else:
                    params += [{'params': [value]}]
        # params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.33)
        num_epochs = args.epochs

        if args.resume!="":
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                model.load_state_dict(checkpoint['model'])
            except RuntimeError:
                print("loading multi gpu model")
                init_weight=OrderedDict()
                for name in checkpoint['model']:
                    init_weight[name[7:]]=checkpoint['model'][name]
                    # init_weight[:checkpoint['model'][name].size()[0]] = checkpoint['model'][name]
                    # delete cls and reg last layer
                model.load_state_dict(init_weight)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            init_epochs = checkpoint['epoch'] + 1
            print("the training process from epoch{}...".format(init_epochs))
            if args.test_only:
                print("Test only:")
                with open(results_file, "a") as f:
                    coco_info, pro = utils.evaluate(model, val_data_set_loader, metaloader, 2, device=device)
                    # 写入的数据包括coco指标还有loss和learning rate
                    result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")
                val_map.append(coco_info[1])  # pascal mAP
                return

        for epoch in range(init_epochs, num_epochs+5, 1):
            # train for one epoch, printing every 50 iterations
            mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader, metaloader,
                                            device, epoch, print_freq=50,cls_w=args.cls, metabs=args.metabs)
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)

            # update the learning rate
            lr_scheduler.step()

            # evaluate on the test dataset
            coco_info, pro = utils.evaluate(model, val_data_set_loader, metaloader, 2, device=device)

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # pascal mAP

            # save weights
            # 仅保存最后5个epoch的权重
            # if epoch in range(num_epochs+init_epochs)[-10:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "{}/mobile-find-{}.pth".format(args.output_dir, epoch))
            # plot loss and lr curve
            if len(train_loss) != 0 and len(learning_rate) != 0:
                from plot_curve import plot_loss_and_lr
                plot_loss_and_lr(train_loss, learning_rate)

            # plot mAP curve
            if len(val_map) != 0:
                from plot_curve import plot_map
                plot_map(val_map)


            # plot loss and lr curve
            if len(train_loss) != 0 and len(learning_rate) != 0:
                from plot_curve import plot_loss_and_lr
                plot_loss_and_lr(train_loss, learning_rate)

            # plot mAP curve
            if len(val_map) != 0:
                from plot_curve import plot_map
                plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:2', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='/home/user/ws/FSDet/data', help='dataset')
    # 文件保存地址
    parser.add_argument('--output_dir', default='./find_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='./find_weights/mobile-find-19.pth', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # traing phase (1/2)
    parser.add_argument('--phase', default=1, type=int,
                        help='which phase of meta learning, 1: meta train, 2: meta fine tune')
    # split (1/2/3)
    parser.add_argument('--meta_type', default=1, type=int,
                        help='which split of VOC to implement, 1, 2, or 3')
    # shots
    parser.add_argument('--shots', default=10, type=int,
                        help='how many shots in few-shot learning')
    # shots
    parser.add_argument('--meta_train', default=True, type=bool,
                        help='is doing meta training/fine tuning?')
    # 训练的batch size
    parser.add_argument('--bs', default=2, type=int, metavar='N',
                        help='batch size when training.')
    # validation batch size
    parser.add_argument('--bs_v', default=2, type=int, metavar='N',
                        help='batch size when training.')
        # 不训练，仅测试
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    # metadata batch size
    parser.add_argument('--metabs', default=4, type=int, metavar='N',
                        help='batch size when training.')
    # weight of cls loss during training
    parser.add_argument('--cls', default=0.3, type=float,
                        help='weight of cls loss during training')


    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
