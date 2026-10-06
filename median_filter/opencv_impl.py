import cv2

from .common import validate_image, validate_ksize

MAX_KSIZE = 255


def median_opencv(img, ksize):
    validate_image(img)
    ksize = validate_ksize(ksize)
    if ksize > MAX_KSIZE:
        raise ValueError(f"cv2.medianBlur при k > {MAX_KSIZE} считает неверно, получено k={ksize}")
    return cv2.medianBlur(img, ksize).reshape(img.shape)
