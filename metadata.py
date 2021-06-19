# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
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
import collections
from draw_box_utils import draw_box

def vis_data(t):
    im=tf.ToPILImage()(t[0])
    im.show()


class MetaDataset(data.Dataset):

    """Meta Dataset
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val')
        metaclass(string): the class name
        shot(int): the number of instances
    """

    def __init__(self, root, image_sets, metaclass, shots=1, shuffle=False, phase=1):
        self.root = root
        self.image_set = image_sets
        self.metaclass = metaclass
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3
        self.shuffle = shuffle
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.shot_path = open(os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'ImageSets/Main/shots.txt'), 'w')
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, 'VOCdevkit', 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

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
            img_path = os.path.join(self.root,"VOCdevkit", data["folder"], "JPEGImages", data["filename"])
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
                    break

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
                self.shot_path.write(str(img_id[1])+'\n')
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
