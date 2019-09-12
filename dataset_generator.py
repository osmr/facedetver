import os
import re
import cv2
import argparse


def parse_args():
    """
    Create python script parameters.
    """
    parser = argparse.ArgumentParser(
        description='Generate dataset images',
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


def correct_crop_rect(rect,
                      img_w,
                      img_h,
                      x_scales=(1.3, 1.3),
                      y_scales=(1.8, 1.6)):
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
               verbose):
    ff_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_frontalface_alt.xml")
    pf_cascade_file_path = os.path.join(work_dir, cascade_dir, "haarcascade_profileface.xml")
    ff_det_cascade = cv2.CascadeClassifier(ff_cascade_file_path)
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
        input_image_file_path_template = os.path.splitext(input_image_file_name)[0]
        input_image_file_path = os.path.join(input_dir_path, input_image_file_name)

        img = cv2.imread(input_image_file_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_img = None

        ff_det = None
        pf_det = None

        ff_dets = ff_det_cascade.detectMultiScale(
            image=img_gray,
            scaleFactor=1.1,
            minNeighbors=3)
        if len(ff_dets) > 0:
            ff_det = ff_dets[0]
        else:
            pf_dets = pf_det_cascade.detectMultiScale(
                image=img_gray,
                scaleFactor=1.1,
                minNeighbors=3)
            if len(pf_dets) > 0:
                pf_det = pf_dets[0]

        img_h = img.shape[0]
        img_w = img.shape[1]
        if ff_det is not None:
            # x, y, w, h = ff_det
            x, y, w, h = correct_crop_rect(ff_det, img_w, img_h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop_img = img[y:y + h, x:(x + w), :]
        elif pf_det is not None:
            # x, y, w, h = pf_det
            x, y, w, h = correct_crop_rect(pf_det, img_w, img_h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            crop_img = img[y:y + h, x:(x + w), :]

        scale_factor = 1.0
        cv2.imshow(
            winname="img",
            mat=cv2.resize(
                src=img,
                dsize=None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST))

        if crop_img is not None:
            cv2.imshow(
                winname="crop_img",
                mat=cv2.resize(
                    src=crop_img,
                    dsize=None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()

    cascade_dir = "cv_cascades"
    crop_faces(
        work_dir=args.work_dir,
        cascade_dir=cascade_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        verbose=False)
