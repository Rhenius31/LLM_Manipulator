#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2
from cv_bridge import CvBridge

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_from_bag.py <bag.mcap> [rgb_topic] [labels_topic]")
        sys.exit(1)

    bag_path = sys.argv[1]
    rgb_topic = sys.argv[2] if len(sys.argv) > 2 else "/rgbd_camera/image"
    labels_topic = sys.argv[3] if len(sys.argv) > 3 else "/segmentation/labels_map"

    out_root = Path("yolo_dataset")
    out_images = out_root / "images" / "train"
    out_labelmaps = out_root / "labelmaps" / "train"
    ensure_dir(out_images)
    ensure_dir(out_labelmaps)

    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if rgb_topic not in topic_types:
        raise RuntimeError(f"RGB topic not in bag: {rgb_topic}. Available: {list(topic_types.keys())}")
    if labels_topic not in topic_types:
        raise RuntimeError(f"Labels topic not in bag: {labels_topic}. Available: {list(topic_types.keys())}")

    rgb_msg_type = get_message(topic_types[rgb_topic])
    labels_msg_type = get_message(topic_types[labels_topic])

    bridge = CvBridge()

    latest_labels = None
    latest_labels_ns = None

    saved = 0
    seen_rgb = 0
    seen_lbl = 0

    # only pair labels if they are within this age of the rgb frame (ns)
    MAX_LABEL_AGE_NS = int(0.25 * 1e9)  # 250 ms

    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == labels_topic:
            msg = deserialize_message(data, labels_msg_type)
            latest_labels = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            latest_labels_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            seen_lbl += 1

        elif topic == rgb_topic:
            msg = deserialize_message(data, rgb_msg_type)
            rgb_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            seen_rgb += 1

            if latest_labels is None:
                continue

            if latest_labels_ns is not None and (rgb_ns - latest_labels_ns) > MAX_LABEL_AGE_NS:
                continue

            base = f"frame_{saved:06d}"
            cv2.imwrite(str(out_images / f"{base}.jpg"), rgb_bgr)
            cv2.imwrite(str(out_labelmaps / f"{base}.png"), latest_labels)
            saved += 1

    print(f"Done. seen_rgb={seen_rgb}, seen_labels={seen_lbl}, saved_pairs={saved}")
    print("Images:", out_images.resolve())
    print("Labelmaps:", out_labelmaps.resolve())


if __name__ == "__main__":
    main()
