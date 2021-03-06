import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
from train_utils.config import cfg

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .roi_head_meta import RoIHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork


class MetaRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, phase):
        super(MetaRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                                       backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        # Fix blocks
        for p in self.backbone[0].parameters(): p.requires_grad = False
        for p in self.backbone[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 5)
        if cfg.RESNET.FIXED_BLOCKS >= 4:
            for p in self.backbone[-1].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.backbone[6].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.backbone[5].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.backbone[4].parameters(): p.requires_grad = False
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        self.backbone.apply(set_bn_fix)
        # use the layers before avg_pool as backbone, a bit different from FSDet
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        self.phase = phase

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.backbone.eval()
            self.backbone[5].train()
            self.backbone[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.apply(set_bn_eval)

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None, get_pro=False, class_prototype=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]], Optional[int], Optional[Dict[str,Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets[0:-1]:         # ????????????????????????target???boxes????????????????????????
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                          boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        if not get_pro:
            original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
            for img in images[0:-1]:
                val = img.shape[-2:]
                assert len(val) == 2  # ?????????????????????????????????
                original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]
        # prn_data are support images
        # ims are query images
        # prn_target are support bboxes

        if get_pro:
            # during phase 2, before eval, first get class_prototype: Dict{"label": tensor}
            assert (self.training == False) and (self.phase ==2)
            prn_data = images[-1]
            prn_target = targets[-1]
            class_proto = OrderedDict()
            for data, tar in zip(prn_data, prn_target):
                cls_data, cls_target = self.transform([data], [tar])  # ???support?????????????????????
                proto_feat = self.backbone(cls_data.tensors) # c * 1280 * 7 * 7
                proto_feat = OrderedDict([('0', proto_feat)])  # ??????????????????????????????????????????????????????????????????
                proto_boxes = [cls_target[0]["boxes"].to(proto_feat["0"].dtype)]
                class_proto[cls_target[0]["labels"]] = self.roi_heads.box_roi_pool(proto_feat, proto_boxes, cls_data.image_sizes)
            return class_proto
                
        if class_prototype!=None:
            # only true when in seconde phase eval or test, assert only test images are input
            assert (self.training == False) and (self.phase == 2)
            ims = images
            tars=None
            images, targets = self.transform(ims, tars)  # ???query?????????????????????
            features = self.backbone(images.tensors)  # ???????????????backbone???????????????
            if isinstance(features, torch.Tensor):  # ???????????????????????????????????????feature???????????????????????????????????????0???
                features = OrderedDict([('0', features)])  # ??????????????????????????????????????????????????????????????????
            proposals, proposal_losses = self.rpn(images, features, targets)
            proto_in = {"feature":None, "tar": None, "sizes": None, "proto": class_prototype}
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets, proto_in)
        else:
            # for phase 1 training, phase 1 eval, and phase 2 training
            prn_data = images[-1]
            prn_target = targets[-1]
            ims = images[0:-1]
            if len(targets)==1:# val/test stage, no data target, just meta target
                tars=None
            else:
                tars = targets[0:-1]
            images, targets = self.transform(ims, tars)  # ???query?????????????????????
            prn_data, prn_target = self.transform(prn_data, prn_target)  # ???support?????????????????????
            features = self.backbone(images.tensors)  # ???????????????backbone???????????????
            # get proto_feature
            proto_feat = self.backbone(prn_data.tensors) # c * 1280 * 7 * 7
            if isinstance(features, torch.Tensor):  # ???????????????????????????????????????feature???????????????????????????????????????0???
                features = OrderedDict([('0', features)])  # ??????????????????????????????????????????????????????????????????
                proto_feat = OrderedDict([('0', proto_feat)])  # ??????????????????????????????????????????????????????????????????
            # ????????????????????????target????????????rpn???
            # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
            # ??????proposals????????????????????????(x1, y1, x2, y2)??????
            proposals, proposal_losses = self.rpn(images, features, targets)
            proto_in = {"feature":proto_feat, "tar": prn_target, "sizes": prn_data.image_sizes, "proto": None}
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets, proto_in)

        # ???????????????????????????????????????????????????bboxes??????????????????????????????
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class PairwiseCosine(nn.Module):
    def __init__(self, inter_batch=False, dim=-1, eps=1e-8):
        super(PairwiseCosine, self).__init__()
        self.inter_batch, self.dim, self.eps = inter_batch, dim, eps
        self.eqn = 'amd,bnd->abmn' if inter_batch else 'bmd,bnd->bmn'
    def forward(self, x, y):
        x=x.unsqueeze(0)
        y=y.unsqueeze(0)
        xx = torch.sum(x**2, dim=self.dim).unsqueeze(-1) # (A, M, 1)
        yy = torch.sum(y**2, dim=self.dim).unsqueeze(-2) # (B, 1, N)
        if self.inter_batch:
            xx, yy = xx.unsqueeze(1), yy.unsqueeze(0) # (A, 1, M, 1), (1, B, 1, N)
        xy = torch.einsum(self.eqn, x, y) if x.shape[1] > 0 else torch.zeros_like(xx * yy)
        return xy / (xx * yy).clamp(min=self.eps**2).sqrt()


class FindPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, phase):
        super(FindPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, 4)
        self.cos_score = PairwiseCosine()
        self.phase = phase
        # self.reweight = nn.Linear(2*num_classes, num_classes)

    def forward(self, r1, r2, c1, c2, meta_tar):
        # r1: box feature reg
        # r2: proto feature reg
        # c1: box feature cls
        # c2: proto feature cls
        # cls branch
        c1 = c1.flatten(start_dim=1)
        c2 = c2.flatten(start_dim=1)
        if self.phase==1:
            meta_score = self.cls_score(c2)
            data_score = self.cls_score(c1)
        else:
            meta_score = None
            data_score = None
        cos_sim = self.cos_score(c1,c2)
        cos_sim = cos_sim.squeeze(0)
        cls_scores=[data_score, meta_score, cos_sim]
        # reg branch
        p = r1.size()[0]
        n = r2.size()[0]
        assert r1.size()[1]==r2.size()[1]
        d = r1.size()[1]
        r1=r1.view(p,-1,d).expand(p,n,d)
        r2=r2.view(-1,n,d).expand(p,n,d)
        r = r1.add(r2)
        bbox_deltas = self.bbox_pred(r)
        bbox_deltas=bbox_deltas.view(p,-1)
        return cls_scores, bbox_deltas


class Find(MetaRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,      # ?????????resize???????????????????????????????????????
                 image_mean=None, image_std=None,  # ?????????normalize???????????????????????????
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # rpn??????nms??????????????????proposal???(??????score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn??????nms??????????????????proposal???
                 rpn_nms_thresh=0.7,  # rpn?????????nms??????????????????iou??????
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn???????????????????????????????????????????????????
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn????????????????????????????????????????????????????????????????????????
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # ?????????????????????      fast rcnn?????????nms???????????????   ?????????????????????score????????????100?????????
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn???????????????????????????????????????????????????
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn???????????????????????????????????????????????????????????????????????????
                 bbox_reg_weights=None, phase=1, class_prototype=None):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # ??????????????????channels
        out_channels = backbone.out_channels

        # ???anchor???????????????????????????????????????resnet50_fpn???anchor?????????
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # ??????RPN????????????????????????????????????
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # ??????rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # ??????rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # ????????????RPN??????
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        #  Multi-scale RoIAlign pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # ????????????????????????roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN???roi pooling??????????????????????????????????????????
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # ????????????7
            representation_size = 1024
            box_head_r = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )
            box_head_c = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )
            box_head={"reg":box_head_r, "cls":box_head_c}

        # ???box_head????????????????????????
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FindPredictor(
                representation_size,
                num_classes, phase)

        # ???roi pooling, box_head??????box_predictor???????????????
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img, phase)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # ?????????????????????????????????????????????batch???????????????
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        # box_head is for proto feature
        super(Find, self).__init__(backbone, rpn, roi_heads, transform, phase)

