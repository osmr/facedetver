"""
    FDV1 classification dataset.
"""

import os
import math
import numpy as np
import mxnet as mx
import cv2
from mxnet.gluon import Block
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data.vision import transforms
from .dataset_metainfo import DatasetMetaInfo


class FDV1(ImageFolderDataset):
    """
    FDV1 (Face Detection Verifier V1) classification dataset.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/fdv1'
        Path to the folder stored the dataset.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "fdv1"),
                 mode="train",
                 transform=None):
        assert (mode in ("train", "val", "test"))
        root = os.path.join(root, mode)
        super(FDV1, self).__init__(root=root, flag=1, transform=transform)
        self.sample_weights = self.calc_sample_weights()

    def calc_sample_weights(self):
        """
        Calculate sample weights vector for weighted sampler.

        Returns
        -------
        np.array
            Sample weights vector.
        """
        list_labels = [i[1] for i in self.items]
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
        return self.items[idx][0]


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
        self.aug_type = "aug0"
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
            "--aug-type",
            type=str,
            default="aug0",
            help="augmentation type. options are aug0, aug1")
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
        self.aug_type = args.aug_type
        self.do_downscale = args.do_downscale


class ImgAugTransform(Block):
    """
    ImgAug-like transform (geometric, noise, and blur).
    """
    def __init__(self):
        super(ImgAugTransform, self).__init__()
        from imgaug import augmenters as iaa
        from imgaug import parameters as iap
        self.seq = iaa.Sequential(
            children=[
                iaa.Sequential(
                    children=[
                        iaa.Affine(
                            rotate=(-25, 25),
                            order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                            mode="edge",
                            name="Affine"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.AdditiveGaussianNoise(
                                        loc=0,
                                        scale=(0.0, 10.0),
                                        per_channel=0.5,
                                        name="AdditiveGaussianNoise")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.SaltAndPepper(
                                        p=(0, 0.01),
                                        per_channel=0.5,
                                        name="SaltAndPepper"))],
                            random_order=True,
                            name="Noise"),
                        iaa.OneOf(
                            children=[
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.MedianBlur(
                                        k=3,
                                        name="MedianBlur")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.AverageBlur(
                                        k=(2, 4),
                                        name="AverageBlur")),
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.GaussianBlur(
                                        sigma=(0.0, 2.0),
                                        name="GaussianBlur"))],
                            name="Blur"),
                    ],
                    random_order=True,
                    name="MainProcess")])

    def forward(self, x):
        img = x.asnumpy().copy()
        # cv2.imshow(winname="imgA", mat=img)
        img_aug = self.seq.augment_image(img)
        # cv2.imshow(winname="img_augA", mat=img_aug)
        # cv2.waitKey()
        x = mx.nd.array(img_aug, dtype=x.dtype, ctx=x.context)
        return x


class Squaring(Block):
    """
    Squaring image.

    Parameters
    ----------
    size : int
        Size of output image.
    interpolation : int, default 1
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1). The result would be slightly
        different on gpu compared to cpu. OpenCV tend to align center while bilinearResize2D
        use algorithm which aligns corner.
    """
    def __init__(self,
                 size,
                 interpolation=1):
        super(Squaring, self).__init__()
        self._size = size
        self._interpolation = interpolation

    def forward(self, x):
        # img = x.asnumpy().copy()
        # cv2.imshow(winname="img", mat=img)
        h, w, _ = x.shape
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
            x = mx.image.copyMakeBorder(x, top, bottom, left, right, type=cv2.BORDER_REPLICATE)
        x = mx.image.imresize(x, self._size, self._size, interp=self._interpolation)
        # img2 = x.asnumpy().copy()
        # cv2.imshow(winname="img2", mat=img2)
        # cv2.waitKey()
        return x


class DownscaleOnDemand(Block):
    """
    Specific image downscale on demand.

    Parameters
    ----------
    size : int
        Size of output image.
    threshold_size : int
        Threshold for size of input image.
    interpolation : int, default 1
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1). The result would be slightly
        different on gpu compared to cpu. OpenCV tend to align center while bilinearResize2D
        use algorithm which aligns corner.
    """
    def __init__(self,
                 size,
                 threshold_size,
                 interpolation=1):
        super(DownscaleOnDemand, self).__init__()
        self._threshold_size = threshold_size
        self._resize = transforms.Resize(size=size, keep_ratio=True, interpolation=interpolation)

    def forward(self, x):
        h, w, _ = x.shape
        if min(h, w) > self._threshold_size:
            x = self._resize(x)
        return x


def fdv1_train_transform(ds_metainfo,
                         mean_rgb=(0.485, 0.456, 0.406),
                         std_rgb=(0.229, 0.224, 0.225),
                         jitter_param=0.4,
                         lighting_param=0.1):
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
    lighting_param : float
        How much to noise intensity of the image.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    interpolation = 1
    if ds_metainfo.aug_type == "aug0":
        transform_list = []
    elif ds_metainfo.aug_type == "aug1":
        transform_list = [
            ImgAugTransform()
        ]
    else:
        raise RuntimeError("Unknown augmentation type: {}\n".format(ds_metainfo.aug_type))

    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    transform_list += [
        Squaring(
            size=resize_value,
            interpolation=interpolation),
        transforms.RandomResizedCrop(
            size=input_image_size,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            interpolation=interpolation),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ]

    return transforms.Compose(transform_list)


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
    Sequential
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
