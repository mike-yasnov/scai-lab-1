import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

METHOD_STYLE = {
    "opencv":    {"color": SERIES[0], "marker": "o", "label": "OpenCV (cv2.medianBlur)"},
    "naive":     {"color": SERIES[1], "marker": "s", "label": "Наивный, чистый Python"},
    "huang":     {"color": SERIES[2], "marker": "^", "label": "Хуанг, чистый Python"},
    "numpy":     {"color": SERIES[3], "marker": "D", "label": "NumPy (векторизация)"},
    "opencv_1t": {"color": SERIES[4], "marker": "v", "label": "OpenCV, 1 поток"},
}

FILTER_STYLE = {
    "median":   {"color": SERIES[0], "marker": "o", "label": "Медианный"},
    "gaussian": {"color": SERIES[1], "marker": "s", "label": "Гауссов"},
    "box":      {"color": SERIES[2], "marker": "^", "label": "Усредняющий (box)"},
    "noisy":    {"color": MUTED, "marker": None, "label": "Без фильтра", "linestyle": "--"},
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "font.size": 11,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
