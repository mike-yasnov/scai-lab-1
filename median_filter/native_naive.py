from .common import apply_per_channel, pad_replicate


def median_naive_channel(channel, ksize):
    if ksize == 1:
        return [row[:] for row in channel]

    height, width = len(channel), len(channel[0])
    radius = ksize // 2
    mid = ksize * ksize // 2
    padded = pad_replicate(channel, radius)

    result = []
    for y in range(height):
        window_rows = padded[y:y + ksize]
        out_row = [0] * width
        for x in range(width):
            window = []
            for row in window_rows:
                window.extend(row[x:x + ksize])
            window.sort()
            out_row[x] = window[mid]
        result.append(out_row)
    return result


def median_naive(img, ksize):
    return apply_per_channel(img, ksize, median_naive_channel)
