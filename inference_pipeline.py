#!/usr/bin/env python3
"""Simple YOLOv5 detection pipeline for image and folder inference.

This script runs YOLOv5 detection on either:
- one image via --image
- a folder of images via --folder

Annotated images with bounding boxes are saved under the chosen project/name directory.
"""

from yolov5.detect import run as detect_run
import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv5 image and folder inference pipeline")
    parser.add_argument("--weights", type=str, default=str(ROOT /
                        "best.pt"), help="Path to model weights")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--image", type=str, help="Path to a single image")
    source_group.add_argument("--folder", type=str,
                              help="Path to a folder of images")

    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--conf-thres", type=float,
                        default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float,
                        default=0.45, help="NMS IoU threshold")
    parser.add_argument("--project", type=str, default=str(ROOT /
                        "runs" / "infer"), help="Save results to project/name")
    parser.add_argument("--name", type=str, default="exp",
                        help="Save results to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="Reuse the existing output folder")
    parser.add_argument("--device", type=str, default="",
                        help="CUDA device, e.g. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true",
                        help="Use OpenCV DNN for ONNX inference")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save detection labels to *.txt")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save detection results to CSV")
    parser.add_argument("--save-crop", action="store_true",
                        help="Save detection crops")
    parser.add_argument("--view-img", action="store_true",
                        help="Display detection results")
    parser.add_argument("--nosave", action="store_true",
                        help="Do not save annotated images")
    return parser.parse_args()


def run_detection(args):
    source = args.image or args.folder
    imgsz = (args.imgsz, args.imgsz)

    detect_run(
        weights=args.weights,
        source=source,
        imgsz=imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        view_img=args.view_img,
        save_txt=args.save_txt,
        save_format=0,
        save_csv=args.save_csv,
        save_conf=False,
        save_crop=args.save_crop,
        nosave=args.nosave,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        half=args.half,
        dnn=args.dnn,
    )

    base_dir = Path(args.project)
    output_dir = base_dir / args.name
    if not output_dir.exists():
        candidates = sorted([p for p in base_dir.glob(
            f"{args.name}*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        output_dir = candidates[0] if candidates else output_dir

    annotated_files = sorted(
        [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    ) if output_dir.exists() else []

    if annotated_files:
        print("Annotated outputs:")
        for item in annotated_files:
            print(item)
    else:
        print(f"No annotated images were found. Checked: {output_dir}")

    return annotated_files


def main():
    args = parse_args()
    run_detection(args)


if __name__ == "__main__":
    main()
