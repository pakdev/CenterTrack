#%%
from collections import defaultdict
import pathlib
import sys
import json

sys.path.insert(0, "src/lib")

import cv2
import jsbeautifier
import numpy as np
from natsort import natsort

from utils.visualizer import Visualizer
from detector import Detector
from opts import opts


def get_paths(root, ext):
    image_paths = []
    if root.is_dir():
        files = list(root.glob(ext))
        for file in natsort(files, key=lambda x: x.name):
            image_paths.append(pathlib.Path(file))
    else:
        image_paths = [root]

    return image_paths


def get_p(calib):
    return np.reshape(
        np.array(
            calib.split("\n")[0].split(":")[1].strip().split(" "), dtype=np.float32
        ),
        (3, -1),
    )


def main():
    args = opts().init()
    root = pathlib.Path(args.demo)
    detector = Detector(args)
    visualizer = Visualizer(args.dataset)

    image_paths = get_paths(root, "*.png")
    calib_paths = get_paths(root / "calib", "*.txt")
    js_opts = jsbeautifier.default_options()
    js_opts.indent_size = 2

    annotations = []
    img_size = (600, 600)
    bird_bboxes = defaultdict(lambda: [])

    for i in range(669):
        image = cv2.imread(str(image_paths[i]))
        calib = get_p(calib_paths[i].read_text())

        results = detector.run(image)
        bird_view = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255

        for result in results["results"]:
            class_id = result["class"] - 1
            if class_id not in [0, 1, 2]:
                continue

            bird_view = visualizer.get_bird_view(bird_view, result, img_size)

            bird_bbox = visualizer.get_bird_bbox(result, img_size)
            if bird_bbox is not None:
                bird_bboxes[i].append(bird_bbox.tolist())

            bbox = visualizer.get_bbox(result, calib)
            if bbox:
                annotations.append(
                    {
                        "image_id": i,
                        "bbox": [
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2] - bbox[0]),
                            int(bbox[3] - bbox[1]),
                        ],
                        "category_id": int(result["class"] - 1),
                    }
                )

    with open("bird_bboxes.json", "w") as f:
        json_bird_bboxes = jsbeautifier.beautify(json.dumps(bird_bboxes), js_opts)
        f.write(json_bird_bboxes)

    with open("annotations.json", "w") as f:
        json_annotations = jsbeautifier.beautify(json.dumps(annotations), js_opts)
        f.write(json_annotations)


main()

# %%
