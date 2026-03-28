# Models Directory

Place your trained YOLO weights here.

## Files

| File | Description |
|------|-------------|
| `yolov8n.pt` | Auto-downloaded by `ultralytics` on first run (COCO pretrained, ~6 MB) |
| `yolov8_sar.pt` | Your fine-tuned SAR ship model — output of `training/train.py` |

## Switching Models

In `backend/model/yolo_inference.py`, change the `MODEL_PATH` constant:

```python
# Use COCO pretrained (default, no download needed after first run)
MODEL_PATH = "yolov8n.pt"

# Use your fine-tuned SAR model
MODEL_PATH = "../models/yolov8_sar.pt"
```

## Download SSDD Fine-Tuned Weights

After running `training/train.py` on the SSDD dataset, the best weights will be
saved automatically to `models/yolov8_sar.pt`.

See `training/README.md` for dataset setup instructions.
