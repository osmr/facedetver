"""
    Script for removing duplicates (for dataset generation).
"""

import os
import re
import cv2
import argparse
import shutil
from skimage.measure import compare_ssim as ssim


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Generate dataset images (Removing Duplicates)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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


def remove_duplicates(work_dir,
                      input_dir,
                      output_dir):
    """
    Removing duplicates.

    Parameters:
    ----------
    work_dir : str
        Working directory path.
    input_dir : str
        Input images directory name.
    output_dir : str
        Output images directory name.
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

    for image_file_name in image_file_names:
        shutil.copy(
            os.path.join(input_dir_path, image_file_name),
            os.path.join(output_dir_path, image_file_name))

    level = 0.9
    for i in range(len(image_file_names)):
        image_file_name_i = image_file_names[i]
        print("processing image: {}".format(image_file_name_i))
        output_image_file_path_i = os.path.join(output_dir_path, image_file_name_i)
        if not os.path.exists(output_image_file_path_i):
            continue
        img_i = cv2.imread(output_image_file_path_i, cv2.IMREAD_COLOR)

        for j in range(i + 1, len(image_file_names)):
            image_file_name_j = image_file_names[j]
            output_image_file_path_j = os.path.join(output_dir_path, image_file_name_j)
            if not os.path.exists(output_image_file_path_j):
                continue
            img_j = cv2.imread(output_image_file_path_j, cv2.IMREAD_COLOR)
            img_j = cv2.resize(
                src=img_j,
                dsize=img_i.shape[:2][::-1],
                interpolation=cv2.INTER_LINEAR)
            ssim_value = ssim(
                X=img_i,
                Y=img_j,
                multichannel=True)
            if ssim_value > level:
                print("--> remove image: {}".format(output_image_file_path_j))
                os.remove(output_image_file_path_j)


if __name__ == "__main__":
    args = parse_args()

    remove_duplicates(
        work_dir=args.work_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir)
