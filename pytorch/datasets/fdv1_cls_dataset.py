"""
    FDV1 classification dataset.
"""

import os
import math
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
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
        self.sample_weights = self.calc_sample_weights()

    def calc_sample_weights(self):
        """
        Calculate sample weights vector for weighted sampler.

        Returns
        -------
        np.array
            Sample weights vector.
        """
        list_labels = [i[1] for i in self.imgs]
        _, label_counts = np.unique(list_labels, return_counts=True)
        total_label_count = label_counts.sum()
        label_widths = float(total_label_count) / label_counts
        sample_weights = np.array([label_widths[i] for i in list_labels], np.float32)
        sample_weights /= sample_weights.sum()
        return sample_weights

    def get_file_name(self, idx):
        """
        Get file path for particular sample.

        Parameters
        ----------
        idx : int
            Number of sample.

        Returns
        -------
        str
            File path.
        """
        return self.imgs[idx][0]


class FDV1MetaInfo(DatasetMetaInfo):
    """
    Descriptor of FDV1 dataset.
    """
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
        self.test_metric_capts = ["Test.Err", "Test.F1", "Test.MCC"]
        self.test_metric_names = ["Top1Error", "F1", "MCC"]
        self.test_metric_extra_kwargs = [
            {"name": "err"},
            {"name": "f1", "average": "micro"},
            {"name": "mcc", "average": "micro"}]
        self.saver_acc_ind = 0
        self.train_transform = fdv1_train_transform
        self.val_transform = fdv1_val_transform
        self.test_transform = fdv1_val_transform
        self.ml_type = "imgcls"
        self.do_downscale = False

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        """
        Create python script parameters (for FDV1 dataset metainfo).

        Parameters:
        ----------
        parser : ArgumentParser
            ArgumentParser instance.
        work_dir_path : str
            Path to working directory.
        """
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
        parser.add_argument(
            "--do-downscale",
            action="store_true",
            help="do force downscale big images")

    def update(self,
               args):
        """
        Update FDV1 dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(FDV1MetaInfo, self).update(args)
        self.input_image_size = (args.input_size, args.input_size)
        self.resize_inv_factor = args.resize_inv_factor
        self.do_downscale = args.do_downscale


class Squaring(object):
    """
    Squaring image.

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
        # img_cv = np.array(img)
        # cv2.imshow(winname="img", mat=img_cv)
        w, h = img.size
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
        # img_cv2 = np.array(img)
        # cv2.imshow(winname="img2", mat=img_cv2)
        # cv2.waitKey()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class DownscaleOnDemand(object):
    """
    Specific image downscale on demand.

    Parameters
    ----------
    size : int
        Size of output image.
    threshold_size : int
        Threshold for size of input image.
    interpolation : int, default 2
        Interpolation method for resizing.
    """
    def __init__(self,
                 size,
                 threshold_size,
                 interpolation=Image.BILINEAR):
        self.size = size
        self.threshold_size = threshold_size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if min(h, w) > self.threshold_size:
            img = F.resize(img, self.size, self.interpolation)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


def fdv1_train_transform(ds_metainfo,
                         mean_rgb=(0.485, 0.456, 0.406),
                         std_rgb=(0.229, 0.224, 0.225),
                         jitter_param=0.4):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        FDV1 dataset metainfo.
    mean_rgb : tuple of 3 float
        Mean of RGB channels in the dataset.
    std_rgb : tuple of 3 float
        STD of RGB channels in the dataset.
    jitter_param : float
        How much to jitter values.

    Returns
    -------
    Compose
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return transforms.Compose([
        Squaring(size=resize_value),
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
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        FDV1 dataset metainfo.
    mean_rgb : tuple of 3 float
        Mean of RGB channels in the dataset.
    std_rgb : tuple of 3 float
        STD of RGB channels in the dataset.

    Returns
    -------
    Compose
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    if ds_metainfo.do_downscale:
        transform_list = [DownscaleOnDemand(size=150, threshold_size=256)]
    else:
        transform_list = []
    transform_list += [
        Squaring(size=resize_value),
        transforms.CenterCrop(size=input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ]
    return transforms.Compose(transform_list)


def calc_val_resize_value(input_image_size=(224, 224),
                          resize_inv_factor=0.875):
    """
    Calculate image resize value for validation subset.

    Parameters:
    ----------
    input_image_size : tuple of 2 int
        Main script arguments.
    resize_inv_factor : float
        Resize inverted factor.

    Returns
    -------
    int
        Resize value.
    """
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))
    return resize_value
