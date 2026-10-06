import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from median_filter import METHODS, add_salt_pepper, median_adaptive
from median_filter.opencv_impl import MAX_KSIZE as OPENCV_MAX_KSIZE

WINDOW = "Median filter"
MAX_RADIUS = 15


def run_filter(img, ksize, method):
    start = time.perf_counter()
    result = METHODS[method](img, ksize)
    return result, time.perf_counter() - start


def with_suffix(path, suffix):
    path = Path(path)
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def save_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Не удалось сохранить {path}")
    print(f"Сохранено: {path}")


def draw_label(img, text, org):
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_preview(img, ksize, method):
    result, elapsed = run_filter(img, ksize, method)
    left, right = img.copy(), result
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    draw_label(left, "input", (10, 25))
    draw_label(right, f"{method}, k={ksize}: {elapsed * 1000:.1f} ms", (10, 25))
    return np.hstack([left, right])


def run_interactive(img, ksize, method, snapshot_dir):
    names = list(METHODS)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("radius (k=2r+1)", WINDOW, min(ksize // 2, MAX_RADIUS), MAX_RADIUS, lambda _: None)
    cv2.createTrackbar("method", WINDOW, names.index(method), len(names) - 1, lambda _: None)
    print("Ползунки: радиус ядра и метод (" + ", ".join(f"{i}={n}" for i, n in enumerate(names)) + ")")
    print("s — сохранить кадр, q/Esc — выход")

    state, frame = None, None
    while True:
        radius = cv2.getTrackbarPos("radius (k=2r+1)", WINDOW)
        method_idx = cv2.getTrackbarPos("method", WINDOW)
        if (radius, method_idx) != state:
            state = (radius, method_idx)
            frame = render_preview(img, 2 * radius + 1, names[method_idx])
            cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            save_image(Path(snapshot_dir) / f"interactive_k{2 * radius + 1}_{names[method_idx]}.png", frame)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def compare_all(img, ksize, output):
    reference_method = "opencv" if ksize <= OPENCV_MAX_KSIZE else "numpy"
    reference, _ = run_filter(img, ksize, reference_method)
    print(f"{'метод':<8} {'время, мс':>12} {'совпадает с ' + reference_method:>20}")
    for method in METHODS:
        if method == "opencv" and ksize > OPENCV_MAX_KSIZE:
            print(f"{method:<8} пропущен: cv2.medianBlur при k > {OPENCV_MAX_KSIZE} считает неверно")
            continue
        result, elapsed = run_filter(img, ksize, method)
        same = np.array_equal(result, reference)
        print(f"{method:<8} {elapsed * 1000:>12.2f} {'да' if same else 'НЕТ':>20}")
        if output:
            save_image(with_suffix(output, f"_{method}"), result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Медианный фильтр с переменным размером ядра")
    parser.add_argument("input", help="путь к исходному изображению")
    parser.add_argument("-o", "--output", help="куда сохранить результат")
    parser.add_argument("-k", "--ksize", type=int, default=5, help="размер ядра, нечётное число >= 1 (по умолчанию 5)")
    parser.add_argument("-m", "--method", default="opencv", choices=[*METHODS, "all"],
                        help="реализация фильтра; all — запустить все и сравнить (по умолчанию opencv)")
    parser.add_argument("--gray", action="store_true", help="читать изображение в оттенках серого")
    parser.add_argument("--noise", type=float, metavar="DENSITY",
                        help="перед фильтрацией добавить шум «соль-перец» с этой плотностью (0..1)")
    parser.add_argument("--seed", type=int, default=0, help="зерно генератора шума")
    parser.add_argument("--adaptive", action="store_true",
                        help="адаптивный медианный фильтр (чистый Python); -k задаёт наибольший размер окна")
    parser.add_argument("--interactive", action="store_true", help="окно с ползунками размера ядра и метода")
    parser.add_argument("--snapshot-dir", default="output", help="куда сохранять кадры интерактивного режима (по умолчанию output)")
    return parser.parse_args(argv)


def check_args(args):
    if args.ksize < 1 or args.ksize % 2 == 0:
        return f"размер ядра должен быть нечётным и не меньше 1, получено {args.ksize}"
    if args.noise is not None and not 0 <= args.noise <= 1:
        return f"плотность шума должна быть от 0 до 1, получено {args.noise}"
    if args.seed < 0:
        return f"зерно шума должно быть неотрицательным, получено {args.seed}"
    if args.output and not cv2.haveImageWriter(args.output):
        return f"не получится сохранить {args.output}: укажите расширение изображения, например .png"
    if args.adaptive and (args.interactive or args.method != "opencv"):
        return "--adaptive нельзя сочетать с --interactive и -m"
    if args.interactive and args.output:
        return "в интерактивном режиме -o не используется, кадр сохраняет клавиша s в --snapshot-dir"
    if args.method == "opencv" and not args.adaptive and args.ksize > OPENCV_MAX_KSIZE:
        return f"cv2.medianBlur при k > {OPENCV_MAX_KSIZE} считает неверно, выберите -m huang, naive или numpy"
    return None


def main(argv=None):
    args = parse_args(argv)
    error = check_args(args)
    if error:
        sys.exit(f"Ошибка: {error}")

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE if args.gray else cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Ошибка: не удалось прочитать изображение {args.input}")
    if img.dtype != np.uint8:
        sys.exit(f"Ошибка: поддерживаются только 8-битные изображения, у {args.input} тип {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    print(f"Изображение {args.input}: {img.shape}")

    try:
        run(args, img)
    except (OSError, ValueError, cv2.error) as exc:
        sys.exit(f"Ошибка: {exc}")


def run(args, img):
    if args.noise is not None:
        img = add_salt_pepper(img, args.noise, seed=args.seed)
        if args.output:
            save_image(with_suffix(args.output, "_noisy"), img)

    if args.adaptive:
        start = time.perf_counter()
        result = median_adaptive(img, args.ksize)
        print(f"adaptive, окно до {args.ksize}x{args.ksize}: {(time.perf_counter() - start) * 1000:.2f} мс")
        if args.output:
            save_image(args.output, result)
    elif args.interactive:
        method = "opencv" if args.method == "all" else args.method
        run_interactive(img, args.ksize, method, args.snapshot_dir)
    elif args.method == "all":
        compare_all(img, args.ksize, args.output)
    else:
        result, elapsed = run_filter(img, args.ksize, args.method)
        print(f"{args.method}, k={args.ksize}: {elapsed * 1000:.2f} мс")
        if args.output:
            save_image(args.output, result)


if __name__ == "__main__":
    main()
