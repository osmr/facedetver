"""
    FDV1 classification dataset.
"""

import os
import math
from torch import functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from .dataset_metainfo import DatasetMetaInfo


class FDV1(ImageFolder):
    """
    FDV1 (Face Detection Verifier V1) classification dataset.

    Parameters
    ----------
    root : str, default '~/.torch/datasets/fdv1'
        Path to the folder stored the dataset.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".torch", "datasets", "fdv1"),
                 mode="train",
                 transform=None):
        assert (mode in ("train", "val", "test"))
        root = os.path.join(root, mode)
        super(FDV1, self).__init__(root=root, transform=transform)


class FDV1MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(FDV1MetaInfo, self).__init__()
        self.label = "FDV1"
        self.short_label = "fdv1"
        self.root_dir_name = "fdv1"
        self.dataset_class = FDV1
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = 2
        self.input_image_size = (224, 224)
        self.resize_inv_factor = 0.875
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.train_use_weighted_sampler = True
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.test_metric_capts = ["Test.Err"]
        self.test_metric_names = ["Top1Error"]
        self.test_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.train_transform = fdv1_train_transform
        self.val_transform = fdv1_val_transform
        self.test_transform = fdv1_val_transform
        self.ml_type = "imgcls"

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(FDV1MetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--input-size",
            type=int,
            default=self.input_image_size[0],
            help="size of the input for model")
        parser.add_argument(
            "--resize-inv-factor",
            type=float,
            default=self.resize_inv_factor,
            help="inverted ratio for input image crop")

    def update(self,
               args):
        super(FDV1MetaInfo, self).update(args)
        self.input_image_size = (args.input_size, args.input_size)


class ResizeLong(object):
    """
    Specific resize the input PIL Image to the given size.

    Parameters
    ----------
    size : int
        Size of output image.
    interpolation : int, default 2
        Interpolation method for resizing.
    """
    def __init__(self,
                 size,
                 interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # import cv2
        # img_cv = img.asnumpy().copy()
        # cv2.imshow(winname="img", mat=img_cv)
        h, w, _ = img.shape
        if h != w:
            top = 0
            bottom = 0
            left = 0
            right = 0
            if h < w:
                top_extra = (w - h) // 2
                top += top_extra
                bottom += w - h - top_extra
            else:
                left_extra = (h - w) // 2
                left += left_extra
                right += h - w - left_extra
            img = F.pad(img, padding=(left, top, right, bottom), padding_mode="edge")
        img = F.resize(img, self.size, self.interpolation)
        # img_cv2 = img.asnumpy().copy()
        # cv2.imshow(winname="img2", mat=img_cv2)
        # cv2.waitKey()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


def fdv1_train_transform(ds_metainfo,
                         mean_rgb=(0.485, 0.456, 0.406),
                         std_rgb=(0.229, 0.224, 0.225),
                         jitter_param=0.4):
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return transforms.Compose([
        ResizeLong(size=resize_value),
        transforms.RandomResizedCrop(
            size=input_image_size,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


def fdv1_val_transform(ds_metainfo,
                       mean_rgb=(0.485, 0.456, 0.406),
                       std_rgb=(0.229, 0.224, 0.225)):
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return transforms.Compose([
        ResizeLong(size=resize_value),
        transforms.CenterCrop(size=input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


def calc_val_resize_value(input_image_size=(224, 224),
                          resize_inv_factor=0.875):
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))
    return resize_value
