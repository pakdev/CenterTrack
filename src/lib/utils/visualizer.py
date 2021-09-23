import cv2
import numpy as np


import utils.ddd_utils as ddd
from dataset.dataset_factory import get_dataset

COLOR_LIST = color_list = np.array(
    [
        (1.000, 1.000, 1.000),
        (0.850, 0.325, 0.098),
        (0.929, 0.694, 0.125),
        (0.494, 0.184, 0.556),
        (0.466, 0.674, 0.188),
        (0.301, 0.745, 0.933),
        (0.635, 0.078, 0.184),
        (0.300, 0.300, 0.300),
        (0.600, 0.600, 0.600),
        (1.000, 0.000, 0.000),
        (1.000, 0.500, 0.000),
        (0.749, 0.749, 0.000),
        (0.000, 1.000, 0.000),
        (0.000, 0.000, 1.000),
        (0.667, 0.000, 1.000),
        (0.333, 0.333, 0.000),
        (0.333, 0.667, 0.000),
        (0.333, 1.000, 0.000),
        (0.667, 0.333, 0.000),
        (0.667, 0.667, 0.000),
        (0.667, 1.000, 0.000),
        (1.000, 0.333, 0.000),
        (1.000, 0.667, 0.000),
        (1.000, 1.000, 0.000),
        (0.000, 0.333, 0.500),
        (0.000, 0.667, 0.500),
        (0.000, 1.000, 0.500),
        (0.333, 0.000, 0.500),
        (0.333, 0.333, 0.500),
        (0.333, 0.667, 0.500),
        (0.333, 1.000, 0.500),
        (0.667, 0.000, 0.500),
        (0.667, 0.333, 0.500),
        (0.667, 0.667, 0.500),
        (0.667, 1.000, 0.500),
        (1.000, 0.000, 0.500),
        (1.000, 0.333, 0.500),
        (1.000, 0.667, 0.500),
        (1.000, 1.000, 0.500),
        (0.000, 0.333, 1.000),
        (0.000, 0.667, 1.000),
        (0.000, 1.000, 1.000),
        (0.333, 0.000, 1.000),
        (0.333, 0.333, 1.000),
        (0.333, 0.667, 1.000),
        (0.333, 1.000, 1.000),
        (0.667, 0.000, 1.000),
        (0.667, 0.333, 1.000),
        (0.667, 0.667, 1.000),
        (0.667, 1.000, 1.000),
        (1.000, 0.000, 1.000),
        (1.000, 0.333, 1.000),
        (1.000, 0.667, 1.000),
        (0.167, 0.000, 0.000),
        (0.333, 0.000, 0.000),
        (0.500, 0.000, 0.000),
        (0.667, 0.000, 0.000),
        (0.833, 0.000, 0.000),
        (1.000, 0.000, 0.000),
        (0.000, 0.167, 0.000),
        (0.000, 0.333, 0.000),
        (0.000, 0.500, 0.000),
        (0.000, 0.667, 0.000),
        (0.000, 0.833, 0.000),
        (0.000, 1.000, 0.000),
        (0.000, 0.000, 0.000),
        (0.000, 0.000, 0.167),
        (0.000, 0.000, 0.333),
        (0.000, 0.000, 0.500),
        (0.000, 0.000, 0.667),
        (0.000, 0.000, 0.833),
        (0.000, 0.000, 1.000),
        (0.333, 0.000, 0.500),
        (0.143, 0.143, 0.143),
        (0.286, 0.286, 0.286),
        (0.429, 0.429, 0.429),
        (0.571, 0.571, 0.571),
        (0.714, 0.714, 0.714),
        (0.857, 0.857, 0.857),
        (0.000, 0.447, 0.741),
        (0.50, 0.5, 0),
    ]
).astype(np.float32)


class Visualizer:
    def __init__(self, dataset_name) -> None:
        dataset = get_dataset(dataset_name)
        colors = [(color_list[i]).astype(np.uint8) for i in range(len(color_list))]
        while len(colors) < len(dataset.class_name):
            colors = colors + colors[: min(len(colors), len(self.names) - len(colors))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        self.names = dataset.class_name
        self.tracked_items = {}

    def get_bird_bbox(self, result, img_size, visibility_threshold=0.3):
        if result["score"] > visibility_threshold:
            print(self.names[result["class"] - 1])
            dim = result["dim"]
            loc = result["loc"]
            rot_y = result["rot_y"]
            world_size = 100
            rect = ddd.compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
            for k in range(4):
                rect[k] = Visualizer._project_3d_to_bird(
                    world_size, img_size[0], rect[k]
                )
            return rect

    def get_bird_view(self, bird_view, result, img_size, visibility_threshold=0.3):
        if result["score"] > visibility_threshold:
            tracking_id = int(result["tracking_id"])
            if not tracking_id in self.tracked_items:
                self.tracked_items[tracking_id] = Visualizer._get_rand_color()

            color = self.tracked_items[tracking_id]
            rect = self.get_bird_bbox(result, img_size, visibility_threshold)
            # center, size, angle = cv2.minAreaRect(rect)
            # if size[0] < 7 or size[1] < 7:  # or size[0] > 30 or size[1] > 30:

            cv2.polylines(
                bird_view,
                [rect.reshape(-1, 1, 2).astype(np.int32)],
                True,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
            for e in [[0, 1]]:
                t = 4 if e == [0, 1] else 1
                cv2.line(
                    bird_view,
                    (int(rect[e[0]][0]), int(rect[e[0]][1])),
                    (int(rect[e[1]][0]), int(rect[e[1]][1])),
                    (0, 0, 255),
                    t,
                )

    def get_bbox(self, result, calib, visibility_threshold=0.3):
        if result["score"] > visibility_threshold:
            dim = result["dim"]
            loc = result["loc"]
            rot_y = result["rot_y"]
            box_3d = ddd.compute_box_3d(dim, loc, rot_y)
            box_2d = ddd.project_to_image(box_3d, calib)
            bbox = [
                box_2d[:, 0].min(),
                box_2d[:, 1].min(),
                box_2d[:, 0].max(),
                box_2d[:, 1].max(),
            ]

            return bbox

    def get_3d_box_view(
        self, image, calib, results, annotations, i, visibility_threshold=0.3
    ):
        img = image
        for result in results["results"]:
            if result["score"] > visibility_threshold:
                tracking_id = int(result["tracking_id"])
                if not tracking_id in self.tracked_items:
                    self.tracked_items[tracking_id] = Visualizer._get_rand_color()

                color = self.tracked_items[tracking_id]
                dim = result["dim"]
                loc = result["loc"]
                rot_y = result["rot_y"]
                if loc[2] > 1 and int(result["class"]) == 1:
                    box_3d = ddd.compute_box_3d(dim, loc, rot_y)
                    box_2d = ddd.project_to_image(box_3d, calib)
                    # img = ddd.draw_box_3d(
                    #     img, box_2d.astype(np.int32), color, same_color=True
                    # )
                    bbox = [
                        box_2d[:, 0].min(),
                        box_2d[:, 1].min(),
                        box_2d[:, 0].max(),
                        box_2d[:, 1].max(),
                    ]
                    img = self._draw_box_2d(
                        img,
                        color,
                        bbox,
                        result["class"] - 1,
                        result["tracking_id"],
                    )

        return img

    def _draw_box_2d(
        self,
        image,
        color,
        bbox,
        class_id,
        tracking_id,
    ):
        name = f"{self.names[class_id]}{tracking_id}"
        thickness = 2
        font_size = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        name_size = cv2.getTextSize(name, font, font_size, thickness)[0]

        bbox = np.array(bbox, dtype=np.int32)
        cv2.rectangle(
            image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness
        )
        # cv2.rectangle(
        #     image,
        #     (bbox[0], bbox[1] - name_size[1] - thickness),
        #     (bbox[0] + name_size[0], bbox[1]),
        #     color,
        #     -1,
        # )
        # cv2.putText(
        #     image,
        #     name,
        #     (bbox[0], bbox[1] - thickness - 1),
        #     font,
        #     font_size,
        #     (0, 0, 0),
        #     thickness=1,
        #     lineType=cv2.LINE_AA,
        # )

        return image

    @staticmethod
    def _project_3d_to_bird(world_size, out_size, point):
        point[0] += world_size / 2
        point[1] = world_size - point[1]
        scaled_point = point * out_size / world_size
        return scaled_point.astype(np.int32)

    @staticmethod
    def _get_rand_color():
        c = ((np.random.random((3)) * 0.6 + 0.2) * 255).astype(np.int32).tolist()
        return c
