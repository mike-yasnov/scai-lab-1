from pathlib import Path

import cv2
from skimage import data

DATA_DIR = Path(__file__).parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    images = {
        "camera.png": data.camera(),
        "astronaut.png": cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR),
        "coffee.png": cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR),
    }
    for name, img in images.items():
        path = DATA_DIR / name
        cv2.imwrite(str(path), img)
        print(f"{path}: {img.shape}")


if __name__ == "__main__":
    main()
