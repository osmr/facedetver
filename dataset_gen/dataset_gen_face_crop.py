"""
    Script for face cropping (for dataset generation).
"""

import os
import re
import cv2
import argparse


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Generate dataset images (Face cropping)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--big-crop",
        dest="big_crop",
        help="Big crop (0/1)",
        default=0,
        type=int)
    parser.add_argument(
        "--unumber",
        dest="unumber",
        help="Unique number for output file names",
        default=7,
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


def correct_crop_rect(rect,
                      img_w,
                      img_h,
                      x_scales=(1.3, 1.3),
                      y_scales=(1.8, 1.6)):
    """
    Correcting of cropping rect.

    Parameters:
    ----------
    rect : tuple of 4 int
        Rectangle in CV format.
    img_w : int
        Image width.
    img_h : int
        Image height.
    x_scales : tuple of 2 int
        Percents of crop extending along x-axis.
    y_scales : tuple of 2 int
        Percents of crop extending along y-axis.

    Returns
    -------
    tuple of 4 int
        Rectangle in CV format.
    """
    x, y, w, h = rect
    c = (x + w // 2, y + h // 2)
    x0 = max(0, c[0] - int(0.5 * x_scales[0] * w))
    y0 = max(0, c[1] - int(0.5 * y_scales[0] * h))
    x1 = min(img_w - 1, c[0] + int(0.5 * x_scales[1] * w))
    y1 = min(img_h - 1, c[1] + int(0.5 * y_scales[1] * h))
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def crop_faces(work_dir,
               cascade_dir,
               input_dir,
               output_dir,
               x_scales,
               y_scales,
               unumber,
               verbose):
    """
    Face cropping in images.

    Parameters:
    ----------
    work_dir : str
        Working directory path.
    cascade_dir : str
        CV cascades directory name.
    input_dir : str
        Input images directory name.
    output_dir : str
        Output images directory name.
    x_scales : tuple of 2 int
        Percents of crop extending along x-axis.
    y_scales : tuple of 2 int
        Percents of crop extending along y-axis.
    unumber : int
        Unique number for output file names.
    verbose : bool
        Whether visualize something.
    """
    assert (verbose is not None)

    ff_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_frontalface_alt.xml")
    ff2_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_frontalface_default.xml")
    ff3_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_frontalface_alt2.xml")
    ff4_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_frontalface_alt_tree.xml")
    pf_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_profileface.xml")
    ff_det_cascade = cv2.CascadeClassifier(ff_cascade_file_path)
    ff2_det_cascade = cv2.CascadeClassifier(ff2_cascade_file_path)
    ff3_det_cascade = cv2.CascadeClassifier(ff3_cascade_file_path)
    ff4_det_cascade = cv2.CascadeClassifier(ff4_cascade_file_path)
    pf_det_cascade = cv2.CascadeClassifier(pf_cascade_file_path)

    input_dir_path = os.path.join(work_dir, input_dir)
    if not os.path.exists(input_dir_path):
        raise Exception("Input dir doesn't exist: {}".format(input_dir_path))

    output_dir_path = os.path.join(work_dir, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    input_image_file_names = os.listdir(input_dir_path)
    input_image_file_names = [n for n in input_image_file_names if os.path.isfile(os.path.join(input_dir_path, n))]
    input_image_file_names.sort(key=lambda var: ["{:10}".format(int(x)) if
                                                 x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    for input_image_file_name in input_image_file_names:
        print("processing image: {}".format(input_image_file_name))
        input_image_file_path = os.path.join(input_dir_path, input_image_file_name)

        img = cv2.imread(input_image_file_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        det = None
        ff_dets = ff_det_cascade.detectMultiScale(
            image=img_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(75, 75))
        if len(ff_dets) > 0:
            det = ff_dets[0]
        else:
            pf_dets = pf_det_cascade.detectMultiScale(
                image=img_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(75, 75))
            if len(pf_dets) > 0:
                det = pf_dets[0]
            else:
                ff2_dets = ff2_det_cascade.detectMultiScale(
                    image=img_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(75, 75))
                if len(ff2_dets) > 0:
                    det = ff2_dets[0]
                else:
                    ff3_dets = ff3_det_cascade.detectMultiScale(
                        image=img_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(75, 75))
                    if len(ff3_dets) > 0:
                        det = ff3_dets[0]
                    else:
                        ff4_dets = ff4_det_cascade.detectMultiScale(
                            image=img_gray,
                            scaleFactor=1.1,
                            minNeighbors=3,
                            minSize=(75, 75))
                        if len(ff4_dets) > 0:
                            det = ff4_dets[0]

        if det is not None:
            img_h = img.shape[0]
            img_w = img.shape[1]
            x, y, w, h = correct_crop_rect(det, img_w, img_h, x_scales, y_scales)
            crop_img = img[y:y + h, x:(x + w), :]

            input_image_file_path_template = os.path.splitext(input_image_file_name)[0]
            output_image_file_path = os.path.join(output_dir_path, "{}-cr{}.jpg".format(
                input_image_file_path_template, unumber))
            cv2.imwrite(output_image_file_path, crop_img)


if __name__ == "__main__":
    args = parse_args()

    x_scales = (1.3, 1.3) if args.big_crop == 1 else (1.15, 1.15)
    y_scales = (1.8, 1.6) if args.big_crop == 1 else (1.6, 1.3)

    cascade_dir = "cv_cascades"
    crop_faces(
        work_dir=args.work_dir,
        cascade_dir=cascade_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        x_scales=x_scales,
        y_scales=y_scales,
        unumber=args.unumber,
        verbose=False)
