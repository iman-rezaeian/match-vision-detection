"""Phase 3A — Fine-tune YOLOv8 on SoccerTrack fisheye data.

Converts SoccerTrack annotations to YOLO format and fine-tunes
YOLOv8s on the domain-specific fisheye soccer images.

Usage:
    python soccer_pipeline_tests/finetune_yolo.py \
        --data data/soccertrack/wide_view/ \
        --epochs 30 \
        --model yolov8s.pt
"""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.soccertrack_loader import load_soccertrack_annotations, find_video_file, find_matching_annotation


def convert_soccertrack_to_yolo(data_dir: str, output_dir: str,
                                 max_frames: int = 500) -> str:
    """
    Convert SoccerTrack annotations + video frames to YOLO training format.

    Creates:
        output_dir/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml

    Returns: path to dataset.yaml
    """
    data_path = Path(data_dir)
    out = Path(output_dir)

    # Create directory structure
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Find video and annotation
    video_path = find_video_file(data_path)
    if video_path is None:
        print("ERROR: No video found")
        return None

    ann_path = find_matching_annotation(video_path, data_path)
    if ann_path is None:
        print("ERROR: No annotation found")
        return None

    print(f"Video: {video_path}")
    print(f"Annotations: {ann_path}")

    # Load annotations
    annotations = load_soccertrack_annotations(str(ann_path))
    print(f"Loaded {len(annotations)} annotated frames")

    # Extract frames and convert labels
    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames uniformly
    frame_ids = sorted(annotations.keys())
    if len(frame_ids) > max_frames:
        indices = np.linspace(0, len(frame_ids) - 1, max_frames, dtype=int)
        frame_ids = [frame_ids[i] for i in indices]

    # 80/20 train/val split
    split_idx = int(len(frame_ids) * 0.8)
    train_frames = frame_ids[:split_idx]
    val_frames = frame_ids[split_idx:]

    print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")

    saved = 0
    for split_name, frame_list in [("train", train_frames), ("val", val_frames)]:
        for fid in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                continue

            bboxes = annotations.get(fid, [])
            if not bboxes:
                continue

            # Save image
            img_name = f"frame_{fid:06d}.jpg"
            cv2.imwrite(str(out / "images" / split_name / img_name), frame)

            # Convert bboxes to YOLO format: class x_center y_center width height (normalized)
            label_name = f"frame_{fid:06d}.txt"
            with open(out / "labels" / split_name / label_name, "w") as f:
                for (x1, y1, x2, y2) in bboxes:
                    cx = ((x1 + x2) / 2) / frame_w
                    cy = ((y1 + y2) / 2) / frame_h
                    w = (x2 - x1) / frame_w
                    h = (y2 - y1) / frame_h

                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            saved += 1

    cap.release()
    print(f"Saved {saved} frame/label pairs")

    # Create dataset.yaml
    yaml_path = out / "dataset.yaml"
    yaml_content = f"""# SoccerTrack Fisheye Dataset
path: {out.resolve()}
train: images/train
val: images/val

nc: 1
names: ['person']
"""
    yaml_path.write_text(yaml_content)
    print(f"Dataset config: {yaml_path}")

    return str(yaml_path)


def finetune(dataset_yaml: str, model_name: str = "yolov8s.pt",
             epochs: int = 30, imgsz: int = 1280,
             output_dir: str = "runs/finetune") -> str:
    """
    Fine-tune YOLOv8 on the converted dataset.

    Returns: path to best.pt weights
    """
    from ultralytics import YOLO

    model = YOLO(model_name)

    print(f"\nStarting fine-tuning:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Dataset: {dataset_yaml}")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=4,  # Conservative for MacBook
        device="mps",  # Apple Silicon GPU
        project=output_dir,
        name="soccertrack_finetune",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        # Augmentation settings for fisheye
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,  # Reduced since images are already wide
        mixup=0.1,
    )

    best_path = Path(output_dir) / "soccertrack_finetune" / "weights" / "best.pt"
    if best_path.exists():
        print(f"\n✅ Fine-tuned model saved: {best_path}")
        return str(best_path)
    else:
        # Try to find it
        for p in Path(output_dir).rglob("best.pt"):
            print(f"\n✅ Fine-tuned model saved: {p}")
            return str(p)

    print("⚠️  Could not find best.pt")
    return model_name


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on SoccerTrack data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to SoccerTrack wide_view directory")
    parser.add_argument("--output", type=str, default="finetune_data",
                        help="Output directory for converted data")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--max_frames", type=int, default=500,
                        help="Max frames to extract for training")
    parser.add_argument("--skip_convert", action="store_true",
                        help="Skip data conversion (use existing)")
    args = parser.parse_args()

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Step 1: Convert data
    yaml_path = str(Path(args.output) / "dataset.yaml")
    if not args.skip_convert or not Path(yaml_path).exists():
        yaml_path = convert_soccertrack_to_yolo(args.data, args.output, args.max_frames)
        if yaml_path is None:
            sys.exit(1)

    # Step 2: Fine-tune
    best_model = finetune(yaml_path, args.model, args.epochs)
    print(f"\nDone. Use fine-tuned model with:")
    print(f"  python test_pipeline.py --model {best_model}")


if __name__ == "__main__":
    main()
