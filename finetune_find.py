import os
import datetime
import pickle

import torch
import torchvision

import transforms
from network_files import Find, AnchorsGenerator
from backbone import MobileNetV2, vgg, resnet101
from finetune_data import MetaDataset, FtDataSet
from my_dataset import VOCDataSet
from train_utils.config import cfg
from train_utils import train_eval_utils_meta as utils
from collections import OrderedDict


def create_model(num_classes, phase):
    # https://download.pytorch.org/models/vgg16-397923af.pth
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512

    backbone = resnet101()
    print("Loading pretrained weights from %s" % ("./backbone/resnet101.pth"))
    state_dict = torch.load("./backbone/resnet101.pth")
    backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})
    backbone.out_channels = 2048

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = Find(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, phase=phase)

    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "find_r/fine_results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 检查保存权重文件夹是否存在
    assert os.path.exists(args.resume)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    #dataset config
    VOC_root = args.data_path  # VOCdevkit
    shots = args.shots
    if args.meta_type == 1:  #  use the first sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_FIRST
    if args.meta_type == 2:  #  use the second sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_SECOND
    if args.meta_type == 3:  #  use the third sets of all classes
        metaclass = cfg.TRAIN.ALLCLASSES_THIRD
    img_set = [('2007', 'trainval')]
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # for baseline metadata is used to generate shots.txt
    metadata = MetaDataset(VOC_root, img_set, metaclass, shots)
    metaloader = torch.utils.data.DataLoader(metadata, batch_size=1,
                                                shuffle=False, num_workers=0, pin_memory=True)

    train_data_set = FtDataSet(VOC_root, metaclass, shots)
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.bs
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=FtDataSet.collate_fn)

    # load validation data set the same, 2012+2007 val.txt
    batch_size = args.bs_v
    val_data_set = VOCDataSet(VOC_root, metaclass, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)
    # create model num_classes equal background + 20 classes
    model = create_model(len(metaclass)+1, 2)
    model.to(device)
    new_state_dict = model.state_dict()
    print("loading checkpoint %s" % (args.resume))
    checkpoint = torch.load(args.resume)

    # delete cls layer from trained find
    for name in list(checkpoint['model'].keys()):
        if ("roi_heads.box_predictor.cls_score" in name):
            del checkpoint['model'][name]
    # load params to model
    new_state_dict.update(checkpoint['model'])
    model.load_state_dict(new_state_dict)
    
    # unfreeze weights of the last layers, others freeze
    for name, parameter in model.named_parameters():
        if ("roi_heads.box_predictor.bbox_pred" in name) or ("roi_heads.box_head_c.fc7" in name) or ("roi_heads.box_head_r.fc7" in name):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs, 1):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader, metaloader,
                                              device, epoch, print_freq=50,cls_w=args.cls, metabs=args.metabs)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        if epoch == range(args.epochs)[-1]:
        # evaluate on the test dataset of last 2 epochs
            coco_info, class_proto_dict = utils.evaluate(model, val_data_set_loader, metaloader, 2, device=device)
            
            if class_proto_dict != None:
                # save class_prototype for test:
                # mean_class_attentions = {k: sum(v) / len(v) for k, v in class_attentions.items()}
                save_path = os.path.join(args.output_dir, 'meta_type_{}'.format(args.meta_type))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(os.path.join(save_path,
                                    str(epoch) + '_shots_' + str(shots) + '_mean_class_attentions.pkl'), 'wb') as f:
                    pickle.dump(class_proto_dict, f, pickle.HIGHEST_PROTOCOL)
                print('save ' + str(args.shots) + ' mean classes attentions done!')

            # write into txt
            with open(results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            val_map.append(coco_info[1])  # pascal mAP

        # save weights
        # 仅保存最后2个epoch的权重
        if epoch in range(args.epochs)[-2:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./fine_find_weight/resnet101-find-{}-{}cls-{}shots.pth".format(epoch,len(metaclass),args.shots))

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
    parser.add_argument('--device', default='cuda:1', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='./', help='dataset')
    # 文件保存地址
    parser.add_argument('--output_dir', default='./fine_find_weight/', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    # split (1/2/3)
    parser.add_argument('--meta_type', default=1, type=int,
                        help='which split of VOC to implement, 1, 2, or 3')
    # shots
    parser.add_argument('--shots', default=1, type=int,
                        help='how many shots in few-shot learning')
    # 训练的batch size
    parser.add_argument('--bs', default=2, type=int, metavar='N',
                        help='batch size when training.')
    # validation batch size
    parser.add_argument('--bs_v', default=1, type=int, metavar='N',
                        help='batch size when training.')
        # metadata batch size
    parser.add_argument('--metabs', default=2, type=int, metavar='N',
                        help='batch size when training.')
    # weight of cls loss during training
    parser.add_argument('--cls', default=0.5, type=float,
                        help='weight of cls loss during training')
    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
