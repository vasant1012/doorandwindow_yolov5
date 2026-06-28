"""
inference.py

YOLOv5 Inference Pipeline

Author: Vasant
"""

import os
import cv2
import torch


class FloorPlanInference:
    def __init__(
        self, weights_path, conf_threshold=0.25, iou_threshold=0.45, device=""
    ):

        self.model = torch.hub.load(
            "yolov5", "custom", path=weights_path, source="local"
        )

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

        if device != "":
            self.model.to(device)

    def predict(self, image_path, save_path):

        results = self.model(image_path)

        detections = results.pandas().xyxy[0]

        results.render()

        annotated = results.ims[0]

        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        os.makedirs("predictions", exist_ok=True)

        cv2.imwrite(save_path, annotated)

        object_count = (
            detections["name"]
            .value_counts()
            .rename_axis("Object")
            .reset_index(name="Count")
        )

        confidence_summary = (
            detections.groupby("name")
            .agg(
                Count=("name", "count"),
                AverageConfidence=("confidence", "mean"),
                MaximumConfidence=("confidence", "max"),
            )
            .reset_index()
            .rename(columns={"name": "Object"})
        )

        return {
            "annotated_image": save_path,
            "detections": detections,
            "object_count": object_count,
            "confidence_summary": confidence_summary,
        }


if __name__ == "__main__":
    MODEL_PATH = "best.pt"

    IMAGE_PATH = "vasant_images"  # NOQA E501

    IMAGE_NAME = "11.jpg"

    predictor = FloorPlanInference(weights_path=MODEL_PATH)

    output = predictor.predict(
        f"./{IMAGE_PATH}/{IMAGE_NAME}", save_path=f"./predictions/prediction_{IMAGE_NAME}" # NOQA E501
    )

    print("\nDetected Objects\n")
    print(output["object_count"])

    print("\nConfidence Summary\n")
    print(output["confidence_summary"])

    print("\nDetailed Detections\n")
    print(output["detections"])

    print("\nAnnotated image saved to:")
    print(output["annotated_image"])
