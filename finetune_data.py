# --------------------------------------------------------
# --------------------------------------------------------
import os
import os.path
import sys
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as tf
import torch
import random
import numpy as np
from lxml import etree
from draw_box_utils import draw_box
import collections
from transforms import RandomHorizontalFlip as flip
from transforms import ToTensor


class MetaDataset(data.Dataset):

    """Meta Dataset
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val')
        metaclass(string): the class name
        shot(int): the number of instances
    """

    def __init__(self, root, image_sets, metaclass, shots=1, shuffle=False):
        self.root = root
        self.image_set = image_sets
        self.metaclass = metaclass
        # phase 2 , following prior work, collect 3*"shots" all cls images, use only "shot" images as metadata, 3*"shots"*2(flipped) base class and shots*2(flipped) novel class as dataset 
        # self.shots = shots * 3
        self.shots = shots
        self.shuffle = shuffle
        self._annopath = os.path.join('%s', 'Annotations','%s', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages','%s'+'2017', '%s.jpg')
        self.shot_path = open(os.path.join(self.root, 'ImageSets/Main/shots.txt'), 'w')
        self.ids = list()
        for line in open(os.path.join(self.root, 'ImageSets', 'Main', self.image_set + '.txt')):
            self.ids.append((self.root, image_sets, line.strip()))

        self.class_to_idx = dict(zip(self.metaclass, range(1, len(self.metaclass)+1)))  # class to index mapping
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        self.prndata = []
        self.prncls = []
        self.prntarget = []
        print("preparing metadataset...")
        prn_image, prn_target = self.get_prndata()
        # make sure for metadatase the meta class follows the same order
        for i in range(shots):
            cls = []
            target = []
            data = []
            for c in range(1,len(self.metaclass)+1):
                img = prn_image[self.idx_to_class[c]][i]
                img_t = prn_target[self.idx_to_class[c]][i]
                cls.append(c)
                # data.append(imgmask.permute(0, 3, 1, 2).contiguous())
                data.append(img)
                target.append(img_t)
            self.prncls.append(cls)
            self.prndata.append(data)
            self.prntarget.append(target)

    def __getitem__(self, index):
        return self.prndata[index], self.prncls[index], self.prntarget[index]

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def get_prndata(self):
        '''
        collect data for dataset and metaset
        :return: the construct prn input data
        :prn_image: lists of images in shape of (H, W, 3)
        :prn_target: lists of information of image
        '''
        if self.shuffle:
            random.shuffle(self.ids)
        prn_image = collections.defaultdict(list)
        prn_target = collections.defaultdict(list)
        classes = collections.defaultdict(int)
        for cls in self.metaclass:
            classes[cls] = 0
        n=0
        for img_id in self.ids:
            xml_path = self._annopath % img_id
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            img_path = os.path.join(self.root, "JPEGImages", self.image_set+'2017', data["filename"])
            image = Image.open(img_path)
            if image.format != "JPEG":
                raise ValueError("Image '{}' format not JPEG".format(self._imgpath % img_id))
            image = F.to_tensor(image)
            assert "object" in data, "{} lack of object information.".format(xml_path)
            for obj in data["object"]:
                if obj["difficult"]=='1':
                    continue
                name = obj["name"]
                if name not in self.metaclass:
                    continue
                if classes[name] >= self.shots:
                    continue

                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue
                classes[name] += 1
                # convert everything into a torch.Tensor
                boxes = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                labels = torch.as_tensor([self.class_to_idx[name]], dtype=torch.int64)
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                self.shot_path.write(str(img_id[-1])+'\n')
                n+=1
                # if self.transforms is not None:
                #     image, target = self.transforms(image, target)
                prn_image[name].append(image)
                prn_target[name].append(target)
                print("loaded meta data: {:d}/{:d}".format(n,len(classes.keys())*self.shots))
            if len(classes) > 0 and min(classes.values()) == self.shots:
                break
        self.shot_path.close()
        return prn_image, prn_target

    def __len__(self):
        return len(self.prndata)

class FtDataSet(data.Dataset):
    """pre-pare dataset for finetuning, only VOC2007"""
    """Assert MetaDataset is firstly called to generat shots.txt"""
    def __init__(self, root, allclass, shots, txt_name: str = "shots.txt"):
        self.root=root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.allclass=allclass

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root,"train", line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_list in self.xml_list:
            assert os.path.exists(xml_list), "not found '{}' file.".format(xml_list)
        self.class_dict = dict(zip(self.allclass, range(1,len(self.allclass)+1)))  # class to index mapping
        # prepare finetune data: 3*"shots"*2(flipped) base class and shots*2(flipped) novel class

        self.flip = flip(1)
        self.t_t = ToTensor()
        self.prepare_data(self.xml_list, shots)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images and targets are already prpared, load
        image = self.images[idx]
        target = self.targets[idx]
        return image, target

    def prepare_data(self, xml_list, shots):
        self.images=[]
        self.targets=[]
        class_count = collections.defaultdict(int)
        for cls in range(1, len(self.allclass)+1):
            class_count[cls] = 0
        print("Collecting {:d} images from: {:s}".format(len(xml_list),"shots.txt"))
        # collect all images and gt in xml_list
        for idx, xml_path in enumerate(xml_list):
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            img_path = os.path.join(self.root, "JPEGImages", "train2017", data["filename"])
            image = Image.open(img_path)
            if image.format != "JPEG":
                raise ValueError("Image '{}' format not JPEG".format(img_path))
            boxes = []
            labels = []
            iscrowd = []
            assert "object" in data, "{} lack of object information.".format(xml_path)
            for obj in data["object"]:
                if obj['name'] not in self.allclass:
                    continue
                cls_id = self.class_dict[obj["name"]]

                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue
                # novel class: only "shots"
                if class_count[cls_id] < shots:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)
                    iscrowd.append(0)
                    class_count[cls_id] += 1
            if len(labels)==0:# no proper object in this image
                continue
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            image_id = torch.tensor([idx])

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["iscrowd"] = iscrowd
            image, target=self.t_t(image, target)
            self.images.append(image)
            self.targets.append(target)
            # flip images and target
            image_, target_ = self.flip(image, target)
            self.images.append(image_)
            self.targets.append(target_)
        print("After filtering and flipping, together {:d} images for finetune".format(len(self.images)))


    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

class COCODataSet(data.Dataset):
    """读取解析COCO val数据集"""

    def __init__(self, root, allclass, transforms, txt_name: str):
        self.root=root
        self.img_root = os.path.join(self.root, "JPEGImages", "val2017")
        self.annotations_root = os.path.join(self.root, "Annotations", "val")
        self.allclass=allclass

        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_list in self.xml_list:
            assert os.path.exists(xml_list), "not found '{}' file.".format(xml_list)
        
        #merge xml_list and filter classes
        self.class_dict = dict(zip(self.allclass, range(1,len(self.allclass)+1)))  # class to index mapping
        self.flip = flip(1)
        self.t_t = ToTensor()
        self.transforms = transforms
        self.filer_data(self.xml_list)
        self.prepare_data(self.xml_list, False)
        print("use COCO val.txt, total images: {:d}".format(len(self.data["path"])))

    def __len__(self):
        return len(self.data["path"])

    def __getitem__(self, idx):
                # read xml
        xml_path = self.data["path"][idx]
        fl = self.data["flip"][idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.root, "JPEGImages",'val2017', data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            # only calculate object in self.allclass
            name = obj["name"]
            if name not in self.allclass:
                continue
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        image, target = self.t_t(image, target)
        if fl:
            image, target = self.flip(image, target)
        return image, target

    def filer_data(self, xml_list):
        self.xml_list = []
        print("Before filtering: {:d} images".format(len(xml_list)))
        obj_num=[]
        for i in range(len(xml_list)):
            xml = xml_list[i]
            with open(xml) as fid:
                xml_str = fid.read()
            xml_e = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml_e)["annotation"]
            proper = False
            clas=[]
            if "object" not in data:
                continue
            for obj in data["object"]:
                label = obj["name"]
                if label not in clas:
                    clas.append(label)
                if label in self.allclass:
                    proper = True
            if proper:
                self.xml_list.append(xml)
                obj_num.append(len(clas))
        print("After filtering: {:d} images for evaluation".format(len(self.xml_list)))
        return

    def prepare_data(self, xml_list, fl):
        self.data={"path":[],"flip":[]}
        for xml_path in xml_list:
            self.data["path"].append(xml_path)
            self.data["flip"].append(False)
            if fl:
                self.data["path"].append(xml_path)
                self.data["flip"].append(True)
        # collect all images and gt in xml_list

        print("After flipping, together {:d} images for eval".format(len(self.data["path"])))

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            # only calculate object in self.allclass
            name = obj["name"]
            if name not in self.allclass:
                continue
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))