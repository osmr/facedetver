"""
    Script for dataset splitting on train/val/test subsets.
"""

import os
import re
import argparse
import shutil
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
        description="Split Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--train",
        dest="train",
        help="Training subset percent",
        default=0.9725014,
        type=int)
    parser.add_argument(
        "--val",
        dest="val",
        help="Validation subset percent",
        default=0.02151201,
        type=int)
    parser.add_argument(
        "--test",
        dest="test",
        help="Test subset percent",
        default=0.005986559,
        type=int)
    parser.add_argument(
        "--unumber",
        dest="unumber",
        help="Unique number for output file names",
        default=1,
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


def split_ds(work_dir,
             input_dir,
             output_dir,
             train_fract,
             val_fract,
             test_fract,
             unumber):
    """
    Split dataset.

    Parameters:
    ----------
    work_dir : str
        Working directory path.
    input_dir : str
        Input images directory name.
    output_dir : str
        Output images directory name.
    train_fract : float
        Training subset fraction.
    val_fract : float
        Validation subset fraction.
    test_fract : float
        Test subset fraction.
    unumber : int
        Unique number for output file names.
    """
    assert (unumber is not None)
    assert (abs(train_fract + val_fract + test_fract - 1.0) < 1e-3)

    input_dir_path = os.path.join(work_dir, input_dir)
    if not os.path.exists(input_dir_path):
        raise Exception("Input dir doesn't exist: {}".format(input_dir_path))

    output_dir_path = os.path.join(work_dir, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    train_output_dir_path = os.path.join(output_dir_path, "train")
    if not os.path.exists(train_output_dir_path):
        os.makedirs(train_output_dir_path)
    val_output_dir_path = os.path.join(output_dir_path, "val")
    if not os.path.exists(val_output_dir_path):
        os.makedirs(val_output_dir_path)
    test_output_dir_path = os.path.join(output_dir_path, "test")
    if not os.path.exists(test_output_dir_path):
        os.makedirs(test_output_dir_path)

    image_file_names = os.listdir(input_dir_path)
    image_file_names = [n for n in image_file_names if os.path.isfile(os.path.join(input_dir_path, n))]
    image_file_names.sort()
    image_file_names.sort(key=lambda var: ["{:10}".format(int(x)) if
                                           x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    image_count = len(image_file_names)
    index_array = np.arange(image_count)
    np.random.shuffle(index_array)
    train_split_at = int(image_count * train_fract)
    val_split_at = int(image_count * (train_fract + val_fract))
    train_index_array = index_array[:train_split_at]
    val_index_array = index_array[train_split_at:val_split_at]
    test_index_array = index_array[val_split_at:]

    index_arrays = (train_index_array, val_index_array, test_index_array)
    output_dir_paths = (train_output_dir_path, val_output_dir_path, test_output_dir_path)

    for index_array_i, output_dir_path_i in zip(index_arrays, output_dir_paths):
        for i in index_array_i:
            shutil.copy(
                os.path.join(input_dir_path, image_file_names[i]),
                os.path.join(output_dir_path_i, image_file_names[i]))


if __name__ == "__main__":
    args = parse_args()

    split_ds(
        work_dir=args.work_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_fract=args.train,
        val_fract=args.val,
        test_fract=args.test,
        unumber=args.unumber)
