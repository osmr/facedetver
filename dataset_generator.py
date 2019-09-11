import os
import cv2


def parse_args():
    """
    Create python script parameters.
    """

    parser = argparse.ArgumentParser(
        description='Generate dataset images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--workdir',
        dest='working_dir_path',
        help='Path to working directory',
        required=True,
        type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
