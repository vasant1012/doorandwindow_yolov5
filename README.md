# Door and Window Object Detection with YOLOv5

This project trains and runs YOLOv5 for door/window object detection.

## Training

From the project root, train a new model with:

```bash
python3 yolov5/train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data data.yaml \
  --weights yolov5s.pt \
  --project runs/train \
  --name door_window
```

The trained weights will be written to:

```bash
runs/train/door_window/weights/best.pt
```

## Inference

### Single image

```bash
python3 inference_pipeline.py \
  --image path/to/image.jpg \
  --weights best.pt \
  --project runs/infer \
  --name exp
```

### Batch inference on a folder

```bash
python3 inference_pipeline.py \
  --folder path/to/images_dir \
  --weights best.pt \
  --project runs/infer \
  --name exp
```

### Optional flags

- `--view-img` to display the annotated image window
- `--save-txt` to save detection labels as text files
- `--save-crop` to save cropped detections
- `--half` for faster inference on supported GPUs

Annotated results will be saved under the chosen output folder, for example:

```bash
runs/infer/exp
```