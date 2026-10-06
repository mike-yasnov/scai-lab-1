import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .common import validate_image, validate_ksize

CHUNK_BYTES = 32 * 2**20


def median_numpy(img, ksize):
    validate_image(img)
    ksize = validate_ksize(ksize)
    if ksize == 1:
        return img.copy()

    radius = ksize // 2
    mid = ksize * ksize // 2
    pad_width = ((radius, radius), (radius, radius)) + ((0, 0),) * (img.ndim - 2)
    padded = np.pad(img, pad_width, mode="edge")

    height = img.shape[0]
    bytes_per_row = img[0].size * ksize * ksize
    rows_per_chunk = max(1, CHUNK_BYTES // bytes_per_row)

    result = np.empty_like(img)
    for top in range(0, height, rows_per_chunk):
        bottom = min(top + rows_per_chunk, height)
        strip = padded[top:bottom + 2 * radius]
        windows = sliding_window_view(strip, (ksize, ksize), axis=(0, 1))
        flat = windows.reshape(*windows.shape[:-2], ksize * ksize)
        result[top:bottom] = np.partition(flat, mid, axis=-1)[..., mid]
    return result
