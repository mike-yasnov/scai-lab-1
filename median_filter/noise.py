import numpy as np


def add_salt_pepper(img, density, seed=None):
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"Плотность шума должна быть в диапазоне [0, 1], получено {density}")
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    mask = rng.random(img.shape[:2])
    noisy[mask < density / 2] = 0
    noisy[(mask >= density / 2) & (mask < density)] = 255
    return noisy
