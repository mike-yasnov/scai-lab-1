from .common import apply_per_channel, pad_replicate


def median_adaptive_channel(channel, max_ksize):
    if max_ksize == 1:
        return [row[:] for row in channel]

    height, width = len(channel), len(channel[0])
    max_radius = max_ksize // 2
    padded = pad_replicate(channel, max_radius)

    result = []
    for y in range(height):
        cy = y + max_radius
        out_row = [0] * width
        for x in range(width):
            cx = x + max_radius
            center = padded[cy][cx]
            for radius in range(1, max_radius + 1):
                window = []
                for row in padded[cy - radius:cy + radius + 1]:
                    window.extend(row[cx - radius:cx + radius + 1])
                window.sort()
                z_min, z_med, z_max = window[0], window[len(window) // 2], window[-1]
                if z_min < z_med < z_max:
                    out_row[x] = center if z_min < center < z_max else z_med
                    break
            else:
                out_row[x] = z_med
        result.append(out_row)
    return result


def median_adaptive(img, max_ksize):
    return apply_per_channel(img, max_ksize, median_adaptive_channel)
