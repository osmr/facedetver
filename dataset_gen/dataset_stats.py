"""
    Script for dataset statistics calculation.
"""

import os
import re
import cv2
import argparse
from tqdm import tqdm
import numpy as np


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Dataset Statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input_images",
        help="name of input image directory")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "facedetver_data"),
        help="path to working directory")
    args = parser.parse_args()
    return args


def calc_stats(work_dir,
               input_dir):
    """
    Calculate dataset statistics.

    Parameters:
    ----------
    work_dir : str
        Working directory path.
    input_dir : str
        Input images directory name.
    """
    input_dir_path = os.path.join(work_dir, input_dir)
    if not os.path.exists(input_dir_path):
        raise Exception("Input dir doesn't exist: {}".format(input_dir_path))

    image_file_names = os.listdir(input_dir_path)
    image_file_names = [n for n in image_file_names if os.path.isfile(os.path.join(input_dir_path, n))]
    image_file_names.sort()
    image_file_names.sort(key=lambda var: ["{:10}".format(int(x)) if
                                           x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    hx = np.zeros((len(image_file_names), ), np.int32)
    wx = np.zeros((len(image_file_names), ), np.int32)
    for i, image_file_name in enumerate(tqdm(image_file_names)):
        # print("processing image: {}".format(image_file_name))
        image_file_path = os.path.join(input_dir_path, image_file_name)
        img = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        hx[i] = img.shape[0]
        wx[i] = img.shape[1]

    def show_stats(x, measure_name):
        print("{} min/max/mean/median/std: {} / {} / {} / {} / {}".format(
            measure_name, x.min(), x.max(), x.mean(), np.median(x), x.std()))

    show_stats(hx, "Height")
    show_stats(wx, "Width")


if __name__ == "__main__":
    args = parse_args()

    calc_stats(
        work_dir=args.work_dir,
        input_dir=args.input_dir)
