from .common import apply_per_channel, pad_replicate

LEVELS = 256


def median_huang_channel(channel, ksize):
    if ksize == 1:
        return [row[:] for row in channel]

    height, width = len(channel), len(channel[0])
    radius = ksize // 2
    half = ksize * ksize // 2
    padded = pad_replicate(channel, radius)

    result = []
    for y in range(height):
        rows = padded[y:y + ksize]

        hist = [0] * LEVELS
        for row in rows:
            for value in row[:ksize]:
                hist[value] += 1

        median = 0
        lt = 0
        while lt + hist[median] <= half:
            lt += hist[median]
            median += 1

        out_row = [0] * width
        out_row[0] = median

        for x in range(1, width):
            left = x - 1
            right = x + ksize - 1
            for row in rows:
                value = row[left]
                hist[value] -= 1
                if value < median:
                    lt -= 1
                value = row[right]
                hist[value] += 1
                if value < median:
                    lt += 1

            while lt > half:
                median -= 1
                lt -= hist[median]
            while lt + hist[median] <= half:
                lt += hist[median]
                median += 1

            out_row[x] = median
        result.append(out_row)
    return result


def median_huang(img, ksize):
    return apply_per_channel(img, ksize, median_huang_channel)
