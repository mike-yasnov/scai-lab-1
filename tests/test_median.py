from pathlib import Path

import cv2
import numpy as np
import pytest

from main import check_args, parse_args
from median_filter import METHODS, add_salt_pepper, median_adaptive, median_filter, median_opencv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OWN_METHODS = ["naive", "huang", "numpy"]
KSIZES = [1, 3, 5, 7, 9, 11, 15, 21, 31]


def random_image(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def assert_matches_opencv(img, ksize, method):
    expected = cv2.medianBlur(img, ksize).reshape(img.shape)
    actual = METHODS[method](img, ksize)
    assert actual.dtype == np.uint8
    assert actual.shape == img.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", OWN_METHODS)
@pytest.mark.parametrize("ksize", KSIZES)
def test_gray_random(method, ksize):
    assert_matches_opencv(random_image((23, 37)), ksize, method)


@pytest.mark.parametrize("method", OWN_METHODS)
@pytest.mark.parametrize("ksize", KSIZES)
def test_color_random(method, ksize):
    assert_matches_opencv(random_image((17, 19, 3), seed=1), ksize, method)


@pytest.mark.parametrize("method", OWN_METHODS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 12), (12, 1), (2, 3), (5, 4), (9, 7, 1), (9, 7, 4)])
@pytest.mark.parametrize("ksize", [3, 5, 31])
def test_small_and_degenerate_shapes(method, shape, ksize):
    assert_matches_opencv(random_image(shape, seed=2), ksize, method)


@pytest.mark.parametrize("method", OWN_METHODS)
@pytest.mark.parametrize("ksize", [3, 7, 15])
def test_extreme_values(method, ksize):
    img = (random_image((21, 21), seed=3) > 127).astype(np.uint8) * 255
    assert_matches_opencv(img, ksize, method)


@pytest.mark.parametrize("method", OWN_METHODS)
@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_image(method, value):
    img = np.full((10, 13), value, dtype=np.uint8)
    np.testing.assert_array_equal(METHODS[method](img, 5), img)


@pytest.mark.parametrize("method", OWN_METHODS)
@pytest.mark.parametrize("ksize", [3, 9])
def test_real_photo_with_noise(method, ksize):
    path = DATA_DIR / "camera.png"
    if not path.exists():
        pytest.skip("нет data/camera.png, запустите prepare_data.py")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[200:264, 200:264]
    assert_matches_opencv(add_salt_pepper(img, 0.2, seed=0), ksize, method)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("ksize", [0, -3, 2, 4])
def test_rejects_bad_ksize(method, ksize):
    with pytest.raises(ValueError):
        METHODS[method](random_image((5, 5)), ksize)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("ksize", [3.0, "3", True, None])
def test_rejects_non_integer_ksize(method, ksize):
    with pytest.raises(TypeError):
        METHODS[method](random_image((5, 5)), ksize)


@pytest.mark.parametrize("method", list(METHODS))
def test_rejects_bad_images(method):
    func = METHODS[method]
    with pytest.raises(TypeError):
        func(np.zeros((5, 5), dtype=np.float32), 3)
    with pytest.raises(TypeError):
        func([[1, 2], [3, 4]], 3)
    with pytest.raises(ValueError):
        func(np.zeros((5,), dtype=np.uint8), 3)
    with pytest.raises(ValueError):
        func(np.zeros((0, 5), dtype=np.uint8), 3)


def test_input_not_modified():
    img = random_image((15, 15))
    copy = img.copy()
    for func in METHODS.values():
        func(img, 5)
    np.testing.assert_array_equal(img, copy)


def test_dispatcher():
    img = random_image((10, 10))
    np.testing.assert_array_equal(median_filter(img, 3, "huang"), cv2.medianBlur(img, 3))
    with pytest.raises(ValueError):
        median_filter(img, 3, "unknown")


def test_salt_pepper_density():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    noisy = add_salt_pepper(img, 0.2, seed=0)
    changed = np.any(noisy != 128, axis=2)
    assert abs(changed.mean() - 0.2) < 0.01
    assert set(np.unique(noisy[changed])) <= {0, 255}
    assert np.all((noisy[changed] == 0).all(axis=1) | (noisy[changed] == 255).all(axis=1))
    with pytest.raises(ValueError):
        add_salt_pepper(img, 1.5)


def adaptive_reference(img, max_ksize):
    r_max = max_ksize // 2
    padded = np.pad(img, r_max, mode="edge")
    out = np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            center = int(img[y, x])
            for r in range(1, r_max + 1):
                window = padded[y + r_max - r:y + r_max + r + 1, x + r_max - r:x + r_max + r + 1]
                z_min, z_med, z_max = int(window.min()), int(np.median(window)), int(window.max())
                if z_min < z_med < z_max:
                    out[y, x] = center if z_min < center < z_max else z_med
                    break
            else:
                out[y, x] = z_med
    return out


@pytest.mark.parametrize("max_ksize", [3, 5, 7, 11])
@pytest.mark.parametrize("density", [0.1, 0.5])
def test_adaptive_matches_reference(max_ksize, density):
    img = add_salt_pepper(random_image((19, 23), seed=4) // 2 + 64, density, seed=5)
    np.testing.assert_array_equal(median_adaptive(img, max_ksize), adaptive_reference(img, max_ksize))


def test_adaptive_keeps_clean_pixels_and_removes_impulses():
    path = DATA_DIR / "camera.png"
    if not path.exists():
        pytest.skip("нет data/camera.png, запустите prepare_data.py")
    clean = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[200:264, 200:264]
    clean = np.clip(clean, 1, 254)
    noisy = add_salt_pepper(clean, 0.3, seed=0)
    out = median_adaptive(noisy, 7)
    impulses = (noisy == 0) | (noisy == 255)
    assert not ((out == 0) | (out == 255)).any()
    assert np.mean(out[~impulses] == noisy[~impulses]) > 0.9


def test_adaptive_color_and_ksize_one():
    img = random_image((6, 7, 3), seed=6)
    np.testing.assert_array_equal(median_adaptive(img, 1), img)
    out = median_adaptive(img, 5)
    for c in range(3):
        np.testing.assert_array_equal(out[:, :, c], adaptive_reference(img[:, :, c], 5))
    with pytest.raises(ValueError):
        median_adaptive(img, 4)


def test_opencv_rejects_huge_kernel():
    with pytest.raises(ValueError):
        median_opencv(random_image((8, 8)), 257)


@pytest.mark.parametrize("argv, fragment", [
    (["x.png", "-k", "4"], "нечётным"),
    (["x.png", "--noise", "1.5"], "плотность"),
    (["x.png", "--noise", "0.1", "--seed", "-1"], "зерно"),
    (["x.png", "-o", "out.xyz"], "расширение"),
    (["x.png", "--adaptive", "--interactive"], "--adaptive"),
    (["x.png", "--adaptive", "-m", "huang"], "--adaptive"),
    (["x.png", "--interactive", "-o", "out.png"], "интерактивном"),
    (["x.png", "-k", "301"], "k > 255"),
])
def test_cli_rejects_bad_arguments(argv, fragment):
    error = check_args(parse_args(argv))
    assert error is not None and fragment in error


def test_cli_accepts_good_arguments():
    assert check_args(parse_args(["x.png", "-o", "out.png", "-k", "301", "-m", "huang"])) is None
    assert check_args(parse_args(["x.png", "-k", "11", "--adaptive", "--noise", "0.5"])) is None
