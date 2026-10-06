from .native_adaptive import median_adaptive
from .native_huang import median_huang
from .native_naive import median_naive
from .noise import add_salt_pepper
from .numpy_impl import median_numpy
from .opencv_impl import median_opencv

METHODS = {
    "opencv": median_opencv,
    "naive": median_naive,
    "huang": median_huang,
    "numpy": median_numpy,
}

NATIVE_METHODS = ("naive", "huang")


def median_filter(img, ksize, method="opencv"):
    try:
        func = METHODS[method]
    except KeyError:
        raise ValueError(f"Неизвестный метод {method!r}, доступны: {', '.join(METHODS)}") from None
    return func(img, ksize)


__all__ = [
    "METHODS",
    "NATIVE_METHODS",
    "add_salt_pepper",
    "median_adaptive",
    "median_filter",
    "median_huang",
    "median_naive",
    "median_numpy",
    "median_opencv",
]
