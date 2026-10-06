import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import plot_style
import matplotlib.pyplot as plt
from main import render_preview
from median_filter import METHODS, add_salt_pepper, median_adaptive
from plot_style import FILTER_STYLE, INK, INK_SECONDARY, MUTED

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42

OWN_METHODS = ["naive", "huang", "numpy"]
CHECK_KSIZES = [1, 3, 5, 7, 9, 15, 21, 31]

DENSITIES = [0.05, 0.10, 0.20, 0.30, 0.40]
PLOT_KSIZES = [3, 5, 7]
BEST_K_CANDIDATES = [3, 5, 7, 9, 11]
GAUSS_SIGMAS = [0.5 + 0.25 * i for i in range(23)]
BOX_KSIZES = [3, 5, 7, 9, 11, 13, 15]
FILTERS = ["median", "gaussian", "box"]

CAMERA_CROP = (slice(60, 260), slice(150, 350))
ASTRONAUT_CROP = (slice(30, 230), slice(130, 330))
COFFEE_CROP = (slice(150, 350), slice(170, 420))

GRID_COLUMNS = [
    ("Исходник", "clean", 0),
    ("С шумом", "noisy", 0),
    ("Медианный, k=3", "median", 3),
    ("Медианный, k=5", "median", 5),
    ("Гауссов, k=5", "gaussian", 5),
    ("Усредняющий, k=5", "box", 5),
]

SWEEP_KSIZES = [3, 5, 7, 9, 15, 31]
LARGE_KSIZES = [5, 15, 31]

STEP_SHAPE = (64, 128)
STEP_LOW, STEP_HIGH = 60, 190
STEP_EDGE = STEP_SHAPE[1] // 2
STEP_DENSITY = 0.15
STEP_ROW = 32
STEP_KSIZE = 7
STEP_WIDTH_KSIZES = [3, 5, 7]


def load(name):
    img = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать {DATA_DIR / name}")
    return img


def apply_filter(kind, img, ksize):
    if kind == "median":
        return cv2.medianBlur(img, ksize)
    if kind == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0, borderType=cv2.BORDER_REPLICATE)
    if kind == "box":
        return cv2.blur(img, (ksize, ksize), borderType=cv2.BORDER_REPLICATE)
    raise ValueError(f"Неизвестный фильтр {kind!r}")


def psnr(clean, img):
    return peak_signal_noise_ratio(clean, img, data_range=255)


def ssim(clean, img):
    channel_axis = 2 if img.ndim == 3 else None
    return structural_similarity(clean, img, data_range=255, channel_axis=channel_axis)


def filter_label(kind, ksize):
    return f"{FILTER_STYLE[kind]['label']}, k={ksize}"


def show(ax, img, title=None):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)


def caption(ax, text):
    ax.text(0.5, -0.03, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=INK_SECONDARY)


def psnr_caption(clean, img):
    return f"PSNR {psnr(clean, img):.1f} дБ"


def save_figure(fig, name):
    path = RESULTS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  сохранено: {path.relative_to(ROOT)}")


def write_csv(name, rows):
    path = RESULTS_DIR / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  сохранено: {path.relative_to(ROOT)}")


def write_text(name, text):
    path = RESULTS_DIR / name
    path.write_text(text, encoding="utf-8")
    print(f"  сохранено: {path.relative_to(ROOT)}")


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def experiment_correctness():
    rng = np.random.default_rng(SEED)
    images = {
        "camera": load("camera.png"),
        "astronaut": load("astronaut.png"),
        "coffee": load("coffee.png"),
        "random": rng.integers(0, 256, (7, 5), dtype=np.uint8),
    }

    rows = []
    for image_name, img in images.items():
        for method in OWN_METHODS:
            start = time.perf_counter()
            for ksize in CHECK_KSIZES:
                expected = cv2.medianBlur(img, ksize)
                actual = METHODS[method](img, ksize)
                diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                differs = diff > 0 if diff.ndim == 2 else (diff > 0).any(axis=2)
                rows.append({
                    "image": image_name,
                    "shape": "x".join(map(str, img.shape)),
                    "method": method,
                    "ksize": ksize,
                    "max_abs_diff": int(diff.max()),
                    "diff_share": float(differs.mean()),
                })
            print(f"  {image_name:<11} {method:<6} проверено за {time.perf_counter() - start:.1f} с", flush=True)
    write_csv("correctness.csv", rows)

    table = []
    for image_name, img in images.items():
        for method in OWN_METHODS:
            cells = [r["max_abs_diff"] for r in rows if r["image"] == image_name and r["method"] == method]
            table.append([f"{image_name} ({'x'.join(map(str, img.shape))})", method, *cells])
    all_zero = all(r["max_abs_diff"] == 0 for r in rows)
    max_share = max(r["diff_share"] for r in rows)
    text = (
        "# Сверка с cv2.medianBlur\n\n"
        "В ячейке — максимальная абсолютная разница max|наш − OpenCV| по всем пикселям и каналам.\n\n"
        + markdown_table(["Изображение", "Метод", *[f"k={k}" for k in CHECK_KSIZES]], table)
        + f"\n\nВсего сравнений: {len(rows)}. "
        + f"Наибольшая доля отличающихся пикселей: {max_share:.4f}. "
        + ("Все реализации совпадают с OpenCV бит в бит.\n" if all_zero
           else "ЕСТЬ РАСХОЖДЕНИЯ — см. correctness.csv.\n")
    )
    write_text("correctness.md", text)

    return [
        f"E1: {len(rows)} сравнений с cv2.medianBlur, все max|diff| = 0: {'да' if all_zero else 'НЕТ'}, "
        f"наибольшая доля отличающихся пикселей {max_share:.4f}",
    ]


def experiment_denoise():
    clean = load("camera.png")

    metrics = {}
    for density in DENSITIES:
        noisy = add_salt_pepper(clean, density, seed=SEED)
        metrics[(density, "noisy", 0)] = {"psnr": psnr(clean, noisy), "ssim": ssim(clean, noisy)}
        for kind in FILTERS:
            ksizes = BEST_K_CANDIDATES if kind == "median" else PLOT_KSIZES
            for ksize in ksizes:
                out = apply_filter(kind, noisy, ksize)
                metrics[(density, kind, ksize)] = {"psnr": psnr(clean, out), "ssim": ssim(clean, out)}

    rows = [
        {"density": density, "filter": kind, "ksize": ksize or "",
         "psnr_db": round(m["psnr"], 4), "ssim": round(m["ssim"], 6)}
        for (density, kind, ksize), m in metrics.items()
    ]
    write_csv("denoise_metrics.csv", rows)

    plot_denoise(metrics, "psnr", "PSNR, дБ", "PSNR после фильтрации (camera, шум «соль-перец»)",
                 "denoise_psnr.png")
    plot_denoise(metrics, "ssim", "SSIM", "SSIM после фильтрации (camera, шум «соль-перец»)",
                 "denoise_ssim.png")

    table, best_ks, best_ks_ssim = [], [], []
    for density in DENSITIES:
        by_k = {k: metrics[(density, "median", k)] for k in BEST_K_CANDIDATES}
        best = max(by_k, key=lambda k: by_k[k]["psnr"])
        best_by_ssim = max(by_k, key=lambda k: by_k[k]["ssim"])
        best_ks.append(best)
        best_ks_ssim.append(best_by_ssim)
        cells = [f"**{m['psnr']:.2f}**" if k == best else f"{m['psnr']:.2f}" for k, m in by_k.items()]
        table.append([f"{density:.0%}", *cells, best, f"{by_k[best]['psnr']:.2f}",
                      f"{by_k[best]['ssim']:.4f}", best_by_ssim])

    grows = all(a <= b for a, b in zip(best_ks, best_ks[1:]))
    if grows and best_ks[-1] > best_ks[0]:
        verdict = (f"Закономерность подтверждается: с ростом плотности шума лучшее k не убывает "
                   f"({' → '.join(map(str, best_ks))}).")
    else:
        verdict = (f"Закономерность подтверждается не полностью: лучшее k по плотностям — "
                   f"{', '.join(map(str, best_ks))}.")
    k_max = max(best_ks)
    if k_max < BEST_K_CANDIDATES[-1]:
        k_next = BEST_K_CANDIDATES[BEST_K_CANDIDATES.index(k_max) + 1]
        gaps = [metrics[(d, "median", k_max)]["psnr"] - metrics[(d, "median", k_next)]["psnr"]
                for d in (DENSITIES[0], DENSITIES[-1])]
        verdict += (f" Рост слабый: k={k_next} и больше не выигрывает ни при одной плотности, "
                    f"но его отставание от k={k_max} сокращается с {gaps[0]:.1f} дБ при {DENSITIES[0]:.0%} "
                    f"до {gaps[1]:.1f} дБ при {DENSITIES[-1]:.0%}.")
    verdict += f" По SSIM лучшее k: {' → '.join(map(str, best_ks_ssim))}."
    text = (
        "# Лучший размер ядра медианного фильтра\n\n"
        f"camera.png, шум «соль-перец», seed={SEED}. В ячейках k=… — PSNR, дБ; лучший выделен.\n\n"
        + markdown_table(["Плотность шума", *[f"k={k}" for k in BEST_K_CANDIDATES],
                          "Лучшее k (PSNR)", "PSNR, дБ", "SSIM", "Лучшее k (SSIM)"], table)
        + f"\n\n{verdict}\n"
    )
    write_text("denoise_best_k.md", text)
    best_linear = best_linear_filters(clean)

    summary = ["E2: PSNR, дБ / SSIM на camera (k = 3, 5, 7):"]
    for density in DENSITIES:
        noisy = metrics[(density, "noisy", 0)]
        parts = [f"без фильтра {noisy['psnr']:.2f}/{noisy['ssim']:.3f}"]
        for kind in FILTERS:
            values = ", ".join(f"{metrics[(density, kind, k)]['psnr']:.2f}/{metrics[(density, kind, k)]['ssim']:.3f}"
                               for k in PLOT_KSIZES)
            parts.append(f"{kind} [{values}]")
        summary.append(f"    шум {density:>4.0%}: " + "; ".join(parts))
    summary.append("E2: лучшее k медианы по PSNR: "
                   + ", ".join(f"{d:.0%} → k={k}" for d, k in zip(DENSITIES, best_ks))
                   + f" ({'растёт' if grows else 'не монотонно'})")
    summary.append("E2: лучшие линейные фильтры по PSNR: " + best_linear)
    return summary


def best_linear_filters(clean):
    table, brief = [], []
    for density in DENSITIES:
        noisy = add_salt_pepper(clean, density, seed=SEED)
        median_k, median_psnr = max(((k, psnr(clean, cv2.medianBlur(noisy, k))) for k in BEST_K_CANDIDATES),
                                    key=lambda item: item[1])
        gauss_sigma, gauss_psnr = max(
            ((sigma, psnr(clean, cv2.GaussianBlur(noisy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)))
             for sigma in GAUSS_SIGMAS),
            key=lambda item: item[1])
        box_k, box_psnr = max(((k, psnr(clean, apply_filter("box", noisy, k))) for k in BOX_KSIZES),
                              key=lambda item: item[1])
        table.append([f"{density:.0%}", f"k={median_k}: {median_psnr:.2f}", f"σ={gauss_sigma:g}: {gauss_psnr:.2f}",
                      f"k={box_k}: {box_psnr:.2f}", f"{median_psnr - max(gauss_psnr, box_psnr):.1f}"])
        brief.append(f"{density:.0%}: медиана {median_psnr:.2f}, гаусс {gauss_psnr:.2f} (σ={gauss_sigma:g}), "
                     f"box {box_psnr:.2f} (k={box_k})")
    text = (
        "# Медиана против линейных фильтров с наилучшими параметрами\n\n"
        f"camera.png, шум «соль-перец», seed={SEED}. PSNR, дБ, по всему снимку. Для каждой плотности "
        f"параметры каждого фильтра подобраны по PSNR: k медианы из {BEST_K_CANDIDATES}, "
        f"sigma гауссова фильтра от {GAUSS_SIGMAS[0]:g} до {GAUSS_SIGMAS[-1]:g} с шагом 0.25, "
        f"k усредняющего из {BOX_KSIZES}.\n\n"
        + markdown_table(["Плотность шума", "Медианный", "Гауссов", "Усредняющий",
                          "Отрыв медианы, дБ"], table)
        + "\n"
    )
    write_text("denoise_best_linear.md", text)
    return "; ".join(brief)


def plot_denoise(metrics, metric, ylabel, title, filename):
    x = [d * 100 for d in DENSITIES]
    fig, axes = plt.subplots(1, len(PLOT_KSIZES), figsize=(13, 4.8), sharey=True, layout="constrained")
    for ax, ksize in zip(axes, PLOT_KSIZES):
        for kind in FILTERS:
            style = FILTER_STYLE[kind]
            ax.plot(x, [metrics[(d, kind, ksize)][metric] for d in DENSITIES],
                    color=style["color"], marker=style["marker"], label=style["label"])
        style = FILTER_STYLE["noisy"]
        ax.plot(x, [metrics[(d, "noisy", 0)][metric] for d in DENSITIES],
                color=style["color"], linestyle=style["linestyle"], linewidth=1.5, label=style["label"])
        ax.set_title(f"k = {ksize}")
        ax.set_xticks(x)
        ax.set_xlabel("Плотность шума, %")
    axes[0].set_ylabel(ylabel)
    if metric == "ssim":
        axes[0].set_ylim(0, 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    save_figure(fig, filename)


def draw_filter_grid(clean, crop, densities, title, filename):
    n_rows = len(densities)
    fig, axes = plt.subplots(n_rows, len(GRID_COLUMNS), figsize=(15, 2.9 * n_rows + 0.6), squeeze=False,
                             layout="constrained")
    for row, density in enumerate(densities):
        noisy = add_salt_pepper(clean, density, seed=SEED)
        for col, (label, kind, ksize) in enumerate(GRID_COLUMNS):
            if kind == "clean":
                out = clean
            elif kind == "noisy":
                out = noisy
            else:
                out = apply_filter(kind, noisy, ksize)
            ax = axes[row, col]
            show(ax, out[crop], label if row == 0 else None)
            caption(ax, "эталон" if kind == "clean" else psnr_caption(clean[crop], out[crop]))
        axes[row, 0].text(-0.06, 0.5, f"шум {density:.0%}", transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    save_figure(fig, filename)


def experiment_grid():
    draw_filter_grid(load("camera.png"), CAMERA_CROP, [0.10, 0.30],
                     "Фильтры на шуме «соль-перец» (camera, PSNR по показанному фрагменту)",
                     "denoise_grid.png")
    draw_filter_grid(load("astronaut.png"), ASTRONAUT_CROP, [0.20],
                     "Цветное изображение (astronaut, PSNR по показанному фрагменту)",
                     "denoise_grid_color.png")
    return []


def experiment_kernel_sweep():
    clean = load("coffee.png")
    noisy = add_salt_pepper(clean, 0.20, seed=SEED)
    panels = [("Исходник", clean), ("Шум 20%", noisy)]
    panels += [(f"Медианный, k={k}", cv2.medianBlur(noisy, k)) for k in SWEEP_KSIZES]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), layout="constrained")
    psnr_by_k = []
    for ax, (title, img) in zip(axes.flat, panels):
        show(ax, img[COFFEE_CROP], title)
        if img is clean:
            caption(ax, "эталон")
        else:
            caption(ax, psnr_caption(clean[COFFEE_CROP], img[COFFEE_CROP]))
            psnr_by_k.append(f"{title.split(', ')[-1]}: {psnr(clean[COFFEE_CROP], img[COFFEE_CROP]):.1f}")
    fig.suptitle("Медианный фильтр с переменным размером ядра (coffee, шум 20%)",
                 fontsize=13, fontweight="bold")
    save_figure(fig, "kernel_sweep.png")
    return ["E4: PSNR фрагмента coffee, дБ — " + "; ".join(psnr_by_k)]


def make_step():
    img = np.full(STEP_SHAPE, STEP_LOW, dtype=np.uint8)
    img[:, STEP_EDGE:] = STEP_HIGH
    return img


def transition_width(profile):
    profile = np.asarray(profile, dtype=np.float64)
    low = np.median(profile[:STEP_EDGE - 12])
    high = np.median(profile[STEP_EDGE + 12:])
    level10 = low + 0.1 * (high - low)
    level90 = low + 0.9 * (high - low)

    i = STEP_EDGE
    while i > 0 and profile[i] >= level10:
        i -= 1
    x10 = i + (level10 - profile[i]) / (profile[i + 1] - profile[i])

    j = STEP_EDGE - 1
    while j < len(profile) - 1 and profile[j] <= level90:
        j += 1
    x90 = j - 1 + (level90 - profile[j - 1]) / (profile[j] - profile[j - 1])
    return x90 - x10


def row_width(img):
    return float(np.median([transition_width(row) for row in img]))


def mean_profile_width(img):
    return transition_width(img.mean(axis=0))


def experiment_edge():
    clean = make_step()
    noisy = add_salt_pepper(clean, STEP_DENSITY, seed=SEED)
    filtered = {(kind, k): apply_filter(kind, noisy, k) for kind in FILTERS for k in STEP_WIDTH_KSIZES}
    widths = {key: row_width(img) for key, img in filtered.items()}
    mean_widths = {key: mean_profile_width(img) for key, img in filtered.items()}

    median_out = filtered[("median", STEP_KSIZE)]
    shifts = np.argmax(median_out >= (STEP_LOW + STEP_HIGH) / 2, axis=1) - STEP_EDGE
    median_row_widths = [transition_width(row) for row in median_out]

    table = [[FILTER_STYLE[kind]["label"],
              *[f"{widths[(kind, k)]:.2f}" for k in STEP_WIDTH_KSIZES],
              *[f"{mean_widths[(kind, k)]:.2f}" for k in STEP_WIDTH_KSIZES]]
             for kind in FILTERS]
    ks_header = [f"k={k}" for k in STEP_WIDTH_KSIZES]
    text = (
        "# Ширина перехода 10–90 % на ступеньке\n\n"
        f"Ступенька {STEP_LOW} → {STEP_HIGH}, изображение {STEP_SHAPE[0]}x{STEP_SHAPE[1]}, "
        f"шум «соль-перец» {STEP_DENSITY:.0%} (seed={SEED}). Ширина в пикселях. "
        "Уровни 10 % и 90 % отсчитываются от уровней самого профиля вдали от ступеньки.\n\n"
        "- «по строкам» — ширина считается для каждой строки отдельно, в таблице медиана по всем строкам;\n"
        "- «по среднему профилю» — ширина профиля, усреднённого по строкам; сюда входит и неровность "
        "границы (если в разных строках граница стоит в разных столбцах).\n\n"
        + markdown_table(["Фильтр", *[f"по строкам, {h}" for h in ks_header],
                          *[f"по среднему профилю, {h}" for h in ks_header]], table)
        + f"\n\nЧистая ступенька (скачок за один пиксель) даёт {transition_width(clean[0]):.2f} px — "
        "это нижняя граница метрики при линейной интерполяции.\n\n"
        f"После медианы k={STEP_KSIZE} ширина перехода в отдельных строках — от {min(median_row_widths):.2f} "
        f"до {max(median_row_widths):.2f} px, но в {np.count_nonzero(shifts)} из {len(shifts)} строк граница "
        f"сдвинута (не более чем на {np.abs(shifts).max()} px): поэтому ширина по среднему профилю "
        "у медианы немного больше.\n"
    )
    write_text("edge_width.md", text)

    series = [
        ("Исходная ступенька", clean, {"color": INK, "linewidth": 1.2, "linestyle": ":", "zorder": 5}),
        (f"С шумом {STEP_DENSITY:.0%}", noisy, {"color": MUTED, "linewidth": 0.8}),
    ]
    for kind in FILTERS:
        style = FILTER_STYLE[kind]
        label = f"{filter_label(kind, STEP_KSIZE)}: переход {widths[(kind, STEP_KSIZE)]:.1f} px"
        series.append((label, filtered[(kind, STEP_KSIZE)],
                       {"color": style["color"], "marker": style["marker"], "markevery": (2, 4), "markersize": 5}))

    fig = plt.figure(figsize=(14, 6.8), layout="constrained")
    grid = fig.add_gridspec(2, len(series), height_ratios=[1, 2.3])
    image_titles = ["Исходник", f"С шумом {STEP_DENSITY:.0%}",
                    *[filter_label(kind, STEP_KSIZE) for kind in FILTERS]]
    for col, ((_, img, line_kw), title) in enumerate(zip(series, image_titles)):
        ax = fig.add_subplot(grid[0, col])
        show(ax, img, title)
        ax.axhline(STEP_ROW, color=line_kw["color"], linewidth=1.2)

    ax = fig.add_subplot(grid[1, :])
    x = np.arange(STEP_SHAPE[1])
    for label, img, line_kw in series:
        ax.plot(x, img[STEP_ROW], drawstyle="steps-mid", label=label, **line_kw)
    ax.set_xlim(0, STEP_SHAPE[1] - 1)
    ax.set_ylim(-5, 260)
    ax.set_yticks([0, STEP_LOW, STEP_HIGH, 255])
    ax.set_xlabel("Столбец, px")
    ax.set_ylabel("Яркость")
    ax.set_title(f"Профиль яркости вдоль строки {STEP_ROW} (ширина перехода 10–90 % — медиана по строкам)",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Сохранение границы: медиана против линейных фильтров", fontsize=13, fontweight="bold")
    save_figure(fig, "edge_profile.png")

    return [
        "E5: ширина перехода 10–90 % на ступеньке, px, медиана по строкам (k = "
        + ", ".join(map(str, STEP_WIDTH_KSIZES)) + "): "
        + "; ".join(f"{kind} [{', '.join(f'{widths[(kind, k)]:.2f}' for k in STEP_WIDTH_KSIZES)}]"
                    for kind in FILTERS)
        + f"; чистая ступенька {transition_width(clean[0]):.2f}",
        "E5: то же по среднему профилю: "
        + "; ".join(f"{kind} [{', '.join(f'{mean_widths[(kind, k)]:.2f}' for k in STEP_WIDTH_KSIZES)}]"
                    for kind in FILTERS),
    ]


def experiment_large_kernels():
    img = load("astronaut.png")
    mosaic = [["original", *[f"median{k}" for k in LARGE_KSIZES]],
              ["original", *[f"gaussian{k}" for k in LARGE_KSIZES]]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 7.6), layout="constrained")
    show(axd["original"], img, "Исходник")
    for kind in ("median", "gaussian"):
        for ksize in LARGE_KSIZES:
            show(axd[f"{kind}{ksize}"], apply_filter(kind, img, ksize), filter_label(kind, ksize))
    fig.suptitle("Большие ядра: медианный фильтр (сверху) и гауссов (снизу)", fontsize=13, fontweight="bold")
    save_figure(fig, "large_kernels.png")
    return []


def experiment_preview():
    noisy = add_salt_pepper(load("coffee.png"), 0.10, seed=SEED)
    frame = render_preview(noisy, 5, "huang")
    path = RESULTS_DIR / "interactive_preview.png"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Не удалось сохранить {path}")
    print(f"  сохранено: {path.relative_to(ROOT)}")
    return []


ADAPTIVE_DENSITIES = [0.10, 0.30, 0.50, 0.60, 0.70]
ADAPTIVE_MAX_KSIZES = [7, 11]


def experiment_adaptive():
    clean = load("camera.png")
    rows, table = [], []
    for density in ADAPTIVE_DENSITIES:
        noisy = add_salt_pepper(clean, density, seed=SEED)
        fixed = {k: cv2.medianBlur(noisy, k) for k in BEST_K_CANDIDATES}
        best_k = max(fixed, key=lambda k: psnr(clean, fixed[k]))
        cells = [f"k={best_k}: {psnr(clean, fixed[best_k]):.2f} / {ssim(clean, fixed[best_k]):.3f}"]
        rows.append({"density": density, "filter": "median_best_fixed", "ksize": best_k,
                     "psnr_db": round(psnr(clean, fixed[best_k]), 4), "ssim": round(ssim(clean, fixed[best_k]), 6),
                     "seconds": ""})
        for max_k in ADAPTIVE_MAX_KSIZES:
            start = time.perf_counter()
            out = median_adaptive(noisy, max_k)
            seconds = time.perf_counter() - start
            cells.append(f"{psnr(clean, out):.2f} / {ssim(clean, out):.3f}")
            rows.append({"density": density, "filter": "adaptive", "ksize": max_k,
                         "psnr_db": round(psnr(clean, out), 4), "ssim": round(ssim(clean, out), 6),
                         "seconds": round(seconds, 3)})
        table.append([f"{density:.0%}", *cells])
    write_csv("adaptive.csv", rows)
    text = (
        "# Адаптивный медианный фильтр против фиксированного окна\n\n"
        f"camera.png, шум «соль-перец», seed={SEED}. В ячейках PSNR, дБ / SSIM по всему снимку. "
        f"Фиксированное окно — лучшее по PSNR из k = {BEST_K_CANDIDATES} (cv2.medianBlur).\n\n"
        + markdown_table(["Плотность шума", "Медиана, лучшее k",
                          *[f"Адаптивная, окно до {k}×{k}" for k in ADAPTIVE_MAX_KSIZES]], table)
        + "\n"
    )
    write_text("adaptive.md", text)

    density = 0.60
    noisy = add_salt_pepper(clean, density, seed=SEED)
    best_k = max(BEST_K_CANDIDATES, key=lambda k: psnr(clean, cv2.medianBlur(noisy, k)))
    panels = [("Исходник", clean), (f"Шум {density:.0%}", noisy),
              ("Медианный, k=5", cv2.medianBlur(noisy, 5)),
              (f"Медианный, k={best_k} (лучшее)", cv2.medianBlur(noisy, best_k)),
              ("Адаптивный, окно до 11×11", median_adaptive(noisy, 11))]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.6), layout="constrained")
    for ax, (label, img) in zip(axes, panels):
        show(ax, img[CAMERA_CROP], label)
        caption(ax, "эталон" if img is clean else psnr_caption(clean[CAMERA_CROP], img[CAMERA_CROP]))
    fig.suptitle("Адаптивный медианный фильтр на шуме 60 % (camera, PSNR по показанному фрагменту)",
                 fontsize=13, fontweight="bold")
    save_figure(fig, "adaptive_grid.png")

    return ["E8: адаптивная медиана, PSNR/SSIM: " + "; ".join(
        f"{r['density']:.0%} {r['filter']} k={r['ksize']}: {r['psnr_db']:.2f}/{r['ssim']:.3f}" for r in rows)]


EXPERIMENTS = {
    "correctness": experiment_correctness,
    "denoise": experiment_denoise,
    "grid": experiment_grid,
    "sweep": experiment_kernel_sweep,
    "edge": experiment_edge,
    "large": experiment_large_kernels,
    "preview": experiment_preview,
    "adaptive": experiment_adaptive,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Эксперименты по корректности и качеству медианного фильтра")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, metavar="NAME",
                        help="запустить только эти эксперименты: " + ", ".join(EXPERIMENTS))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_style.apply_style()

    summary = []
    total_start = time.perf_counter()
    for name in args.only or EXPERIMENTS:
        print(f"[{name}]")
        start = time.perf_counter()
        summary += EXPERIMENTS[name]()
        print(f"[{name}] готово за {time.perf_counter() - start:.1f} с\n")

    print("=" * 30, "Сводка", "=" * 30)
    for line in summary:
        print(line)
    print(f"Общее время: {time.perf_counter() - total_start:.1f} с")


if __name__ == "__main__":
    main()
