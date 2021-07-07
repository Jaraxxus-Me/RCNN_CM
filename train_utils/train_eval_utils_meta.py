import math
import sys
import time
import os
import pickle

import torch
from torch.autograd import Variable
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, meta_loader, device, epoch,
                    print_freq=50, warmup=False, cls_w=0.3, metabs=4):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    enable_amp = True if "cuda" in device.type else False
    meta_iter = iter(meta_loader)
    n=0
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            prndata, prncls, prntar = next(meta_iter)
        except:
            meta_iter = iter(meta_loader)
            prndata, prncls, prntar = next(meta_iter)
        #images: b*3*W*H
        #prnims: n*3*W*H
        images = list(image.to(device) for image in images)
        prnims = list(Variable(prnim.squeeze(0)).to(device) for prnim in prndata)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        prntar = [{k: v.to(device) for k, v in t.items()} for t in prntar]
        protolabel=[]
        for b in range(len(images)):
            for la in targets[b]["labels"]:
                if la.item() not in protolabel:
                    protolabel.append(la.item())
        protoim=[]
        prototar=[]
        for l in protolabel:
            protoim.append(prnims[l-1])
            prototar.append(prntar[l-1])
            # maximum images to load as meta image
            if len(protoim) >= metabs:
                break
        images.append(protoim)
        targets.append(prototar)
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=enable_amp):
            loss_dict = model(images, targets)
            # reduce weight of cls loss
            loss_dict["loss_classifier"]=sum(cls_ for cls_ in loss_dict["loss_classifier"].values())
            loss_dict["loss_classifier"]=cls_w*loss_dict["loss_classifier"]
            n+=1
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            # 记录训练损失
            mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

            if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, meta_loader, phase, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    meta_iter = iter(meta_loader)
    created=False
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        try:
            prndata, prncls, prntar = next(meta_iter)
        except:
            meta_iter = iter(meta_loader)
            prndata, prncls, prntar = next(meta_iter)

        #images: b*3*W*H
        #prnims: n*3*W*H
        images = list(img.to(device) for img in image)
        prnims = list(Variable(prnim.squeeze(0)).to(device) for prnim in prndata)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        prntar = [{k: v.to(device) for k, v in t.items()} for t in prntar]
        if phase==2:
            if not created:
            # during finetuning, last epoch, eval, save class_prototype 
                print("calculating per class prototype...")   
                class_proto = model([prnims],[prntar],get_pro=True)
                created = True
                print("done!")
            model_time = time.time()
            # offering the model test images and class prototypes
            outputs = model(images, class_prototype = class_proto)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
        else:
            # during phase 1 eval:
            class_proto = None
            protolabel=[]
            for b in range(len(images)):
                for la in targets[b]["labels"]:
                    if la.item() not in protolabel:
                        protolabel.append(la.item())
            protoim=[]
            prototar=[]
            for l in protolabel:
                protoim.append(prnims[l-1])
                prototar.append(prntar[l-1])
                # every image must have a prototype
            images.append(protoim)
            prntar=[prototar]
            # 当使用CPU时，跳过GPU相关指令
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

            model_time = time.time()
            outputs = model(images,prntar)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info, class_proto


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
