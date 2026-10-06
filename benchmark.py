import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from median_filter import METHODS
from plot_style import INK_SECONDARY, METHOD_STYLE, MUTED, SURFACE, apply_style

DATA_DIR = Path(__file__).parent / "data"

BENCH_METHODS = ["naive", "huang", "numpy", "opencv_1t", "opencv"]
SLOW_METHODS = ("naive", "huang")

CROP = 256
KSIZES = [3, 5, 7, 9, 11, 15, 21, 31]
SIDES = [64, 128, 256, 512, 1024]
SIZE_KSIZE = 5
COLOR_KSIZES = [5, 15]
QUICK_KSIZES = [3, 5, 7]
QUICK_SIDES = [64, 128]
ALPHA_RANGE = (5, 31)

FAST_BUDGET_S = 0.5
FAST_MIN_REPEATS = 7
FAST_MAX_REPEATS = 200
SLOW_REPEATS = 3
QUICK_SLOW_REPEATS = 1

CSV_FIELDS = ["experiment", "method", "ksize", "height", "width", "channels", "pixels",
              "repeats", "time_min_s", "time_median_s", "ns_per_pixel"]


def cpu_model():
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "неизвестно"


def describe_environment(quick):
    return {
        "cpu": cpu_model(),
        "cpu_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "quick": quick,
        "date": datetime.now().isoformat(timespec="seconds"),
    }


def enough_repeats(times, repeats):
    if repeats is not None:
        return len(times) >= repeats
    if len(times) < FAST_MIN_REPEATS:
        return False
    return sum(times) >= FAST_BUDGET_S or len(times) >= FAST_MAX_REPEATS


def measure(func, img, ksize, repeats=None):
    func(img, ksize)
    times = []
    while not enough_repeats(times, repeats):
        start = time.perf_counter()
        func(img, ksize)
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times), len(times)


@contextmanager
def opencv_single_thread():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def check_result(method, result, reference, img, ksize):
    if not np.array_equal(result, reference):
        raise RuntimeError(f"{method}: результат для k={ksize}, {img.shape} не совпадает с cv2.medianBlur")


def bench_point(experiment, method, img, ksize, slow_repeats):
    func = METHODS["opencv" if method == "opencv_1t" else method]
    repeats = slow_repeats if method in SLOW_METHODS else None
    reference = cv2.medianBlur(img, ksize)
    threads = opencv_single_thread() if method == "opencv_1t" else nullcontext()
    with threads:
        if method != "opencv":
            check_result(method, func(img, ksize), reference, img, ksize)
        t_min, t_median, n = measure(func, img, ksize, repeats)

    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    print(f"  {method:<10} k={ksize:<3} {height}x{width}x{channels:<3} "
          f"{t_median * 1e3:10.3f} мс  (мин. {t_min * 1e3:.3f}, повторов: {n})", flush=True)
    return {
        "experiment": experiment, "method": method, "ksize": ksize,
        "height": height, "width": width, "channels": channels, "pixels": height * width,
        "repeats": n, "time_min_s": t_min, "time_median_s": t_median,
        "ns_per_pixel": t_median / (height * width) * 1e9,
    }


def run_experiment(experiment, cases, slow_repeats):
    rows = []
    for img, ksize in cases:
        for method in BENCH_METHODS:
            rows.append(bench_point(experiment, method, img, ksize, slow_repeats))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранено: {path}")


def load_image(name, flags):
    path = DATA_DIR / name
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать {path}")
    return img


def center_crop(img, size):
    top = (img.shape[0] - size) // 2
    left = (img.shape[1] - size) // 2
    return img[top:top + size, left:left + size].copy()


def resize_square(img, side):
    if side == img.shape[0]:
        return img.copy()
    interpolation = cv2.INTER_AREA if side < img.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(img, (side, side), interpolation=interpolation)


def times_by(rows, method, key="ksize"):
    return {r[key]: r["time_median_s"] for r in rows if r["method"] == method}


def growth_exponent(rows, method):
    times = times_by(rows, method)
    ks = [k for k in sorted(times) if ALPHA_RANGE[0] <= k <= ALPHA_RANGE[1]]
    if len(ks) < 2:
        return None
    slope, _ = np.polyfit(np.log(ks), np.log([times[k] for k in ks]), 1)
    return slope


def naive_huang_crossover(rows):
    naive, huang = times_by(rows, "naive"), times_by(rows, "huang")
    ks = sorted(naive.keys() & huang.keys())
    d = [math.log(naive[k] / huang[k]) for k in ks]
    if d[0] > 0:
        return f"Хуанг быстрее наивного уже при наименьшем k = {ks[0]}."
    for i in range(len(ks) - 1):
        if d[i] <= 0 < d[i + 1]:
            x0, x1 = math.log(ks[i]), math.log(ks[i + 1])
            k_cross = math.exp(x0 - d[i] * (x1 - x0) / (d[i + 1] - d[i]))
            return (f"Хуанг становится быстрее наивного при k ≈ {k_cross:.1f} "
                    f"(между k = {ks[i]} и k = {ks[i + 1]}).")
    return f"Наивный алгоритм быстрее Хуанга во всём диапазоне k = {ks[0]}…{ks[-1]}."


def fit_native_growth(rows):
    def points(method):
        pts = [r for r in rows if r["method"] == method]
        return np.array([r["ksize"] for r in pts], float), np.array([r["ns_per_pixel"] / 1e3 for r in pts])

    k, t = points("naive")
    (a,), *_ = np.linalg.lstsq(k[:, None] ** 2, t, rcond=None)
    k, t = points("huang")
    (b, c), *_ = np.linalg.lstsq(np.column_stack([k, np.ones_like(k)]), t, rcond=None)
    return a, b, c


def method_line(method):
    style = METHOD_STYLE[method]
    return {"color": style["color"], "marker": style["marker"], "label": style["label"],
            "markeredgecolor": SURFACE, "markeredgewidth": 1.2}


def plain_log_ticks(axis):
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def finish(fig, ax, title, subtitle, path, legend_outside=True):
    ax.set_title(title, loc="left", pad=26)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=10, va="bottom")
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    print(f"Сохранено: {path}")


def plot_time_vs_ksize(rows, path):
    ks = sorted(times_by(rows, "opencv"))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in BENCH_METHODS:
        times = times_by(rows, method)
        ax.plot(ks, [times[k] * 1e3 for k in ks], **method_line(method))
    ax.set_yscale("log")
    plain_log_ticks(ax.yaxis)
    ax.set_xticks(ks)
    ax.set_xlabel("Размер ядра k (окно k×k)")
    ax.set_ylabel("Время, мс (лог. шкала)")
    finish(fig, ax, "Время фильтрации от размера ядра",
           f"Серое изображение {CROP}×{CROP}, медиана повторов", path)


def plot_time_vs_size(rows, path):
    sides = sorted(times_by(rows, "opencv", key="width"))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in BENCH_METHODS:
        times = times_by(rows, method, key="width")
        ax.plot([s * s for s in sides], [times[s] * 1e3 for s in sides], **method_line(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_ticks(ax.yaxis)
    ax.set_xticks([s * s for s in sides], labels=[f"{s}×{s}\n{s * s / 1e6:.3g} Мп" for s in sides])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Размер изображения: сторона, px, и мегапиксели (лог. шкала)")
    ax.set_ylabel("Время, мс (лог. шкала)")
    finish(fig, ax, "Время фильтрации от размера изображения",
           f"Серое изображение, k = {SIZE_KSIZE}, медиана повторов", path)


def plot_speedup(rows, path):
    base = times_by(rows, "opencv")
    ks = sorted(base)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in BENCH_METHODS:
        if method == "opencv":
            continue
        times = times_by(rows, method)
        ax.plot(ks, [times[k] / base[k] for k in ks], **method_line(method))
    ax.axhline(1, color=MUTED, linewidth=1, zorder=1)
    ax.set_yscale("log")
    plain_log_ticks(ax.yaxis)
    ax.set_xticks(ks)
    ax.set_xlabel("Размер ядра k (окно k×k)")
    ax.set_ylabel("Во сколько раз OpenCV быстрее (лог. шкала)")
    finish(fig, ax, "Ускорение OpenCV относительно других реализаций",
           f"t(метод) / t(OpenCV), серое {CROP}×{CROP}; выше 1 — OpenCV быстрее", path)


def plot_native_growth(rows, path):
    a, b, c = fit_native_growth(rows)
    all_ks = sorted(times_by(rows, "naive"))
    k_dense = np.linspace(all_ks[0], all_ks[-1], 200)
    fits = {
        "naive": (a * k_dense ** 2, f"МНК: {a:.3g}·k²"),
        "huang": (b * k_dense + c, f"МНК: {b:.3g}·k {'+' if c >= 0 else '−'} {abs(c):.3g}"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, method in zip(axes, SLOW_METHODS):
        pts = [r for r in rows if r["method"] == method]
        ax.plot([r["ksize"] for r in pts], [r["ns_per_pixel"] / 1e3 for r in pts],
                linestyle="none", markersize=8, **method_line(method))
        fit_values, fit_label = fits[method]
        ax.plot(k_dense, fit_values, color=METHOD_STYLE[method]["color"],
                linestyle="--", linewidth=1.2, label=fit_label, zorder=1)
        ax.set_ylim(bottom=0)
        ax.set_xticks(all_ks)
        ax.set_xlabel("Размер ядра k (окно k×k)")
    axes[0].set_ylabel("Время на пиксель, мкс")
    axes[1].legend(loc="upper left")
    finish(fig, axes[0], "Рост времени нативных реализаций",
           f"Серое {CROP}×{CROP}; точки — замеры, пунктир — аппроксимация МНК; шкалы по y у панелей разные",
           path, legend_outside=False)


def fmt(x, digits=3):
    if x == 0 or not math.isfinite(x):
        return str(x)
    decimals = digits - 1 - math.floor(math.log10(abs(x)))
    if decimals < 0:
        return str(int(round(x, decimals)))
    return f"{x:.{decimals}f}"


def md_table(header, body, numbers=True):
    other = "---:|" if numbers else ":---|"
    lines = ["| " + " | ".join(header) + " |", "|:---|" + other * (len(header) - 1)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def label(method):
    return METHOD_STYLE[method]["label"]


def make_summary(env, ksize_rows, size_rows, color_rows, slow_repeats):
    ks = sorted(times_by(ksize_rows, "opencv"))
    sides = sorted(times_by(size_rows, "opencv", key="width"))
    base = times_by(ksize_rows, "opencv")
    others = [m for m in BENCH_METHODS if m != "opencv"]
    out = []

    if env["quick"]:
        out += ["> **Быстрый режим (`--quick`)**: параметры урезаны, числа годятся только для проверки скрипта.", ""]

    out += ["### Стенд", "", md_table(["Параметр", "Значение"], [
        ["Процессор", env["cpu"]],
        ["Логических ядер", str(env["cpu_cores"])],
        ["ОС", env["os"]],
        ["Python", env["python"]],
        ["NumPy", env["numpy"]],
        ["OpenCV", env["opencv"]],
        ["Потоков OpenCV по умолчанию", str(env["opencv_threads"])],
        ["Дата замеров", env["date"]],
    ], numbers=False), ""]

    out += ["### Методика", "",
            "- Время — медиана повторов (минимум тоже есть в CSV); перед замером один прогревочный вызов.",
            f"- OpenCV и NumPy повторяются, пока суммарное время меньше {FAST_BUDGET_S} с: "
            f"от {FAST_MIN_REPEATS} до {FAST_MAX_REPEATS} повторов. "
            f"Реализации на чистом Python: {slow_repeats} {'повтор' if slow_repeats == 1 else 'повтора'}.",
            "- Перед замером результат каждого метода, включая OpenCV в одном потоке, сверяется "
            "с `cv2.medianBlur` в обычном режиме: все совпали бит в бит во всех точках.", ""]

    out += [f"### Время от размера ядра (серое {CROP}×{CROP}), мс", "",
            md_table(["Метод"] + [f"k={k}" for k in ks],
                     [[label(m)] + [fmt(times_by(ksize_rows, m)[k] * 1e3) for k in ks] for m in BENCH_METHODS]),
            ""]

    out += ["### Во сколько раз OpenCV быстрее", "",
            md_table(["Метод"] + [f"k={k}" for k in ks],
                     [[label(m)] + [fmt(times_by(ksize_rows, m)[k] / base[k]) for k in ks] for m in others]),
            ""]

    alpha_ks = [k for k in ks if ALPHA_RANGE[0] <= k <= ALPHA_RANGE[1]]
    alpha_rows = []
    for m in BENCH_METHODS:
        alpha = growth_exponent(ksize_rows, m)
        alpha_rows.append([label(m), "—" if alpha is None else f"{alpha:.2f}"])
    a, b, c = fit_native_growth(ksize_rows)
    out += [f"### Показатель роста time ~ k^α (МНК в лог-лог, k = {', '.join(map(str, alpha_ks))})", "",
            md_table(["Метод", "α"], alpha_rows), "",
            f"Аппроксимация времени на пиксель по всем k: наивный ≈ {fmt(a)}·k² мкс, "
            f"Хуанг ≈ {fmt(b)}·k {'+' if c >= 0 else '−'} {fmt(abs(c))} мкс.", "",
            naive_huang_crossover(ksize_rows), ""]

    out += [f"### Время от размера изображения (k = {SIZE_KSIZE})", "", "Время, мс:", "",
            md_table(["Метод"] + [f"{s}×{s}" for s in sides],
                     [[label(m)] + [fmt(times_by(size_rows, m, key="width")[s] * 1e3) for s in sides]
                      for m in BENCH_METHODS]),
            "", "Время на пиксель, нс:", "",
            md_table(["Метод"] + [f"{s}×{s}" for s in sides],
                     [[label(m)] + [fmt(r["ns_per_pixel"]) for r in size_rows if r["method"] == m]
                      for m in BENCH_METHODS]),
            ""]

    color_body = []
    for k in COLOR_KSIZES:
        for m in BENCH_METHODS:
            t = {r["channels"]: r["time_median_s"] for r in color_rows if r["method"] == m and r["ksize"] == k}
            color_body.append([label(m), str(k), fmt(t[1] * 1e3), fmt(t[3] * 1e3), fmt(t[3] / t[1])])
    out += [f"### Серое против цветного ({CROP}×{CROP})", "",
            md_table(["Метод", "k", "Серое, мс", "Цветное, мс", "Цвет / серое"], color_body), ""]
    return "\n".join(out)


def print_brief(env, ksize_rows, color_rows):
    base = times_by(ksize_rows, "opencv")
    print("\nСводка")
    print(f"  Стенд: {env['cpu']}, {env['cpu_cores']} ядер, OpenCV {env['opencv']} "
          f"({env['opencv_threads']} потоков), NumPy {env['numpy']}, Python {env['python']}")
    print(f"  Во сколько раз OpenCV быстрее (по k = {min(base)}…{max(base)}):")
    for m in BENCH_METHODS:
        if m == "opencv":
            continue
        ratios = [t / base[k] for k, t in times_by(ksize_rows, m).items()]
        print(f"    {m:<10} от {fmt(min(ratios))} до {fmt(max(ratios))} раз")
    alphas = []
    for m in BENCH_METHODS:
        alpha = growth_exponent(ksize_rows, m)
        alphas.append(f"{m} {'—' if alpha is None else f'{alpha:.2f}'}")
    print("  Показатель роста time ~ k^alpha: " + ", ".join(alphas))
    print("  " + naive_huang_crossover(ksize_rows))
    for k in COLOR_KSIZES:
        ratios = []
        for m in BENCH_METHODS:
            t = {r["channels"]: r["time_median_s"] for r in color_rows if r["method"] == m and r["ksize"] == k}
            ratios.append(f"{m} {t[3] / t[1]:.2f}")
        print(f"  Цвет / серое при k = {k}: " + ", ".join(ratios))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение быстродействия реализаций медианного фильтра")
    parser.add_argument("--quick", action="store_true",
                        help="быстрая проверка: меньше k и размеров, медленные методы по одному повтору")
    parser.add_argument("--out", default="results", help="папка для результатов (по умолчанию results)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    started = time.perf_counter()

    env = describe_environment(args.quick)
    (out / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Стенд: {env['cpu']}, {env['cpu_cores']} ядер, {env['os']}")
    print(f"Python {env['python']}, NumPy {env['numpy']}, OpenCV {env['opencv']} ({env['opencv_threads']} потоков)")

    ksizes = QUICK_KSIZES if args.quick else KSIZES
    sides = QUICK_SIDES if args.quick else SIDES
    slow_repeats = QUICK_SLOW_REPEATS if args.quick else SLOW_REPEATS
    camera = load_image("camera.png", cv2.IMREAD_GRAYSCALE)
    astronaut = load_image("astronaut.png", cv2.IMREAD_COLOR)

    print(f"\nЭксперимент 1: время от размера ядра, серое {CROP}x{CROP}")
    crop = center_crop(camera, CROP)
    ksize_rows = run_experiment("ksize", [(crop, k) for k in ksizes], slow_repeats)
    write_csv(ksize_rows, out / "bench_ksize.csv")

    print(f"\nЭксперимент 2: время от размера изображения, k = {SIZE_KSIZE}")
    size_rows = run_experiment("size", [(resize_square(camera, s), SIZE_KSIZE) for s in sides], slow_repeats)
    write_csv(size_rows, out / "bench_size.csv")

    print(f"\nЭксперимент 3: серое против цветного, {CROP}x{CROP}")
    color = center_crop(astronaut, CROP)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    color_rows = run_experiment("color", [(img, k) for k in COLOR_KSIZES for img in (gray, color)], slow_repeats)
    write_csv(color_rows, out / "bench_color.csv")

    print()
    plot_time_vs_ksize(ksize_rows, out / "bench_time_vs_ksize.png")
    plot_time_vs_size(size_rows, out / "bench_time_vs_size.png")
    plot_speedup(ksize_rows, out / "bench_speedup.png")
    plot_native_growth(ksize_rows, out / "bench_native_growth.png")

    summary = make_summary(env, ksize_rows, size_rows, color_rows, slow_repeats)
    (out / "bench_summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"Сохранено: {out / 'bench_summary.md'}")

    print_brief(env, ksize_rows, color_rows)
    print(f"\nПрогон занял {time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    main()
