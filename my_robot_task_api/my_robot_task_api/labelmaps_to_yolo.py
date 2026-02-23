#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np

LABEL_TO_CLASS = {
    1: 0,  # cup
    2: 1,  # box
    3: 2,  # table
    4: 3,  # tray
}

def decode_ids(lbl_bgr: np.ndarray) -> np.ndarray:
    # Your labels_map is grayscale replicated in B,G,R => pick one channel
    return lbl_bgr[:, :, 0].astype(np.int32)

def to_yolo(cls, x, y, w, h, W, H):
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    bw = w / W
    bh = h / H
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def main():
    root = Path("yolo_dataset")
    labelmaps = root / "labelmaps" / "train"
    out_labels = root / "labels" / "train"
    out_labels.mkdir(parents=True, exist_ok=True)

    files = sorted(labelmaps.glob("*.png"))
    if not files:
        raise RuntimeError("No labelmaps found.")

    # quick debug
    sample = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    ids = decode_ids(sample)
    print("Sample unique IDs:", np.unique(ids))

    empty = 0
    for p in files:
        lbl = cv2.imread(str(p), cv2.IMREAD_COLOR)
        ids = decode_ids(lbl)
        H, W = ids.shape

        lines = []
        for label_id, cls_id in LABEL_TO_CLASS.items():
            mask = (ids == label_id).astype(np.uint8) * 255
            if mask.sum() == 0:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 2 or h < 2:
                    continue
                lines.append(to_yolo(cls_id, x, y, w, h, W, H))

        out_txt = out_labels / f"{p.stem}.txt"
        if lines:
            out_txt.write_text("\n".join(lines) + "\n")
        else:
            out_txt.write_text("")
            empty += 1

    print("Done. YOLO labels:", out_labels.resolve())
    print(f"Empty label files: {empty} / {len(files)}")

if __name__ == "__main__":
    main()
