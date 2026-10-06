import numpy as np


def validate_ksize(ksize):
    if isinstance(ksize, bool) or not isinstance(ksize, (int, np.integer)):
        raise TypeError(f"Размер ядра должен быть целым числом, получено {ksize!r}")
    ksize = int(ksize)
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"Размер ядра должен быть нечётным и не меньше 1, получено {ksize}")
    return ksize


def validate_image(img):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Ожидается numpy.ndarray, получено {type(img).__name__}")
    if img.dtype != np.uint8:
        raise TypeError(f"Поддерживается только тип uint8, получено {img.dtype}")
    if img.ndim not in (2, 3):
        raise ValueError(f"Ожидается изображение формы (H, W) или (H, W, C), получено {img.shape}")
    if img.size == 0:
        raise ValueError("Изображение пустое")


def pad_replicate(channel, radius):
    height = len(channel)
    padded = []
    for y in range(-radius, height + radius):
        row = channel[min(max(y, 0), height - 1)]
        padded.append([row[0]] * radius + row + [row[-1]] * radius)
    return padded


def apply_per_channel(img, ksize, channel_filter):
    validate_image(img)
    ksize = validate_ksize(ksize)
    if img.ndim == 2:
        return np.array(channel_filter(img.tolist(), ksize), dtype=np.uint8)
    channels = [
        np.array(channel_filter(img[:, :, c].tolist(), ksize), dtype=np.uint8)
        for c in range(img.shape[2])
    ]
    return np.stack(channels, axis=2)
