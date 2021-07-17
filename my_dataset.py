from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
import collections
from transforms import RandomHorizontalFlip as flip
from transforms import ToTensor


class VOC2007DataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, transforms, txt_name: str = "train.txt"):
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2007")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
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

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

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

class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, allclass, transforms, txt_name: str):
        self.root=os.path.join(voc_root, "VOCdevkit")
        self.root_12 = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.root_07 = os.path.join(voc_root, "VOCdevkit", "VOC2007")
        self.img_root_12 = os.path.join(self.root_12, "JPEGImages")
        self.img_root_07 = os.path.join(self.root_07, "JPEGImages")
        self.annotations_root_12 = os.path.join(self.root_12, "Annotations")
        self.annotations_root_07 = os.path.join(self.root_07, "Annotations")
        self.allclass=allclass

        # read train.txt or val.txt file
        if "val" in txt_name:
            name_07=txt_name
            name_12=txt_name
        else:
            names = txt_name.split("+")
            name_07 = names[0]+".txt"
            name_12 = names[1]+".txt"
        txt_path_12 = os.path.join(self.root_12, "ImageSets", "Main", name_12)
        txt_path_07 = os.path.join(self.root_07, "ImageSets", "Main", name_07)
        assert os.path.exists(txt_path_12), "not found {} file.".format(name_12)
        assert os.path.exists(txt_path_07), "not found {} file.".format(name_07)

        with open(txt_path_12) as read:
            self.xml_list_12 = [os.path.join(self.annotations_root_12, line.strip() + ".xml")
                             for line in read.readlines()]
        with open(txt_path_07) as read:
            self.xml_list_07 = [os.path.join(self.annotations_root_07, line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list_12) > 0, "in '{}' file does not find any information.".format(txt_path_12)
        for xml_list_12 in self.xml_list_12:
            assert os.path.exists(xml_list_12), "not found '{}' file.".format(xml_list_12)
        assert len(self.xml_list_07) > 0, "in '{}' file does not find any information.".format(txt_path_07)
        for xml_list_07 in self.xml_list_07:
            assert os.path.exists(xml_list_07), "not found '{}' file.".format(xml_list_07)
        
        #merge xml_list and filter classes
        self.class_dict = dict(zip(self.allclass, range(1,len(self.allclass)+1)))  # class to index mapping
        self.flip = flip(1)
        self.t_t = ToTensor()
        self.transforms = transforms
        li = self.xml_list_07+self.xml_list_12
        li=li[0:100]
        self.filer_data(li)
        if "val" in txt_name:
            self.prepare_data(self.xml_list, False)
            print("use VOC 2007+2012 val.txt, total images: {:d}".format(len(self.data["path"])))
        else:
            self.prepare_data(self.xml_list, True)
            print("prepared VOC 2007+2012, total images: {:d}".format(len(self.data["path"])))

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
        img_path = os.path.join(self.root, data["folder"], "JPEGImages", data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
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
            proper = True
            clas=[]
            for obj in data["object"]:
                label = obj["name"]
                if label not in clas:
                    clas.append(label)
                if label not in self.allclass:
                    proper = False
            if proper:
                self.xml_list.append(xml)
                obj_num.append(len(clas))
        print("After filtering: {:d} images".format(len(self.xml_list)))
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

        print("After flipping, together {:d} images for base train / val".format(len(self.data["path"])))

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

# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random

# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)

# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }

# # load train data set
# train_data_set = VOCDataSet('/home/li/CMU_RISS/FSDet/data', data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
