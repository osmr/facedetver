import os
import re
import cv2
import argparse
import random
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def parse_args():
    """
    Create python script parameters.
    """
    parser = argparse.ArgumentParser(
        description="Generate dataset images (Hard Augmentation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--count",
        dest="count",
        help="Maximal count of output images",
        default=20000,
        type=int)
    parser.add_argument(
        "--unumber",
        dest="unumber",
        help="Unique number for output file names",
        default=0,
        type=int)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input_images",
        help="name of input image directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_images",
        help="name of output image directory")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "facedetver_data"),
        help="path to working directory")
    args = parser.parse_args()
    return args


def create_augmneter():
    """
    Augmentator declaration.

    Returns
    -------
    iaa.Sequential
        imgaug-augmenter.
    """
    def random_size_crop(src):
        h, w, _ = src.shape
        src_area = h * w
        area = (0.15, 0.25)
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        for _ in range(10):
            target_area = random.uniform(area[0], area[1]) * src_area
            new_ratio = random.uniform(*ratio)
            new_w = int(round(np.sqrt(target_area * new_ratio)))
            new_h = int(round(np.sqrt(target_area / new_ratio)))
            if random.random() < 0.5:
                new_h, new_w = new_w, new_h
            if new_w <= w and new_h <= h:
                x0 = random.randint(0, w - new_w)
                y0 = random.randint(0, h - new_h)
                return src[y0:(y0 + new_h), x0:(x0 + new_w)]
        new_w = int(round(0.3 * w))
        new_h = int(round(0.3 * h))
        x0 = int((w - new_w) / 2)
        y0 = int((h - new_h) / 2)
        return src[y0:(y0 + new_h), x0:(x0 + new_w)]

    def rand_crop(images, random_state, parents, hooks):
        images[0] = random_size_crop(images[0])
        return images

    return iaa.Sequential(
        children=[
            iaa.Affine(
                rotate=(-25.0, 25.0),
                order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                mode="edge"),
            iaa.OneOf(
                children=[
                    iaa.Rot90((1, 3), keep_size=False),
                    iaa.Grayscale(alpha=1.0),
                    iaa.AddToHueAndSaturation((25, 255)),
                    iaa.Lambda(func_images=rand_crop)
                ])
        ])


def augment(work_dir,
            input_dir,
            output_dir,
            max_count,
            unumber):
    """
    Do augmentation on images.

    Parameters:
    ----------
    work_dir : str
        Working directory path.
    input_dir : str
        Input images directory name.
    output_dir : str
        Output images directory name.
    unumber : int
        Unique number for output file names.
    """
    input_dir_path = os.path.join(work_dir, input_dir)
    if not os.path.exists(input_dir_path):
        raise Exception("Input dir doesn't exist: {}".format(input_dir_path))

    output_dir_path = os.path.join(work_dir, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    image_file_names = os.listdir(input_dir_path)
    image_file_names = [n for n in image_file_names if os.path.isfile(os.path.join(input_dir_path, n))]
    image_file_names.sort()
    image_file_names.sort(key=lambda var: ["{:10}".format(int(x)) if
                                           x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    aug_seq = create_augmneter()

    index_array = np.arange(len(image_file_names))
    np.random.shuffle(index_array)
    index_array = index_array[:max_count]
    for i in index_array:
        image_file_name = image_file_names[i]
        print("processing image: {}".format(image_file_name))
        input_image_file_path = os.path.join(input_dir_path, image_file_name)
        img = cv2.imread(input_image_file_path, cv2.IMREAD_COLOR)
        img_aug = aug_seq.augment_image(img)

        image_file_path_template = os.path.splitext(image_file_name)[0]
        output_image_file_path = os.path.join(output_dir_path, "{}-au{}.jpg".format(image_file_path_template, unumber))
        cv2.imwrite(output_image_file_path, img_aug)


if __name__ == "__main__":
    args = parse_args()

    augment(
        work_dir=args.work_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_count=args.count,
        unumber=args.unumber)
